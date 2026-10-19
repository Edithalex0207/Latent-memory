#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""后台落档参考实现：聊天一直在一个会话里进行，记忆由后台进程按天整理、写进 Latent，聊天的人不用管。

  python3 background_archive.py run      # 检查一次：够门槛就整理一段写进去，不够就退出（配给 cron）
  python3 background_archive.py watch    # 常驻：每 --interval 秒检查一次
  python3 background_archive.py status   # 看游标、上次成功、连续失败次数
  python3 background_archive.py flush    # 不看门槛，把游标之后的全部整理掉（续接裁上下文之前用）

一次「落档」做的事：
  1. 读对话来源（Claude Code 会话 jsonl，或通用 jsonl），取游标之后的消息，按本地自然日切出最老那一天。
  2. 够门槛才动：攒满 --min-messages 条，或最老一条已经等了 --stale-minutes 分钟，或那一天已经过去。
  3. 起一次性模型子进程（默认 `claude -p --tools ""`，**不给任何工具**）写当天的整理稿，再自审一遍。
     输出必须是约定形状（首行「# 对话整理 · 日期」、末行「当下状态：…」），不合形状带纠正提示重来
     一次，还不合就不落盘、游标不动、计一次失败。模型判断不值得写就回 PASS，那是正常出口。
  4. 通过 HTTP 调 `latent_append` 写进记忆库（走服务端完整校验；--facts 时一并带上事实条目）。
  5. **写成功才推进游标**。连续失败到 --notify-after 次，跑一次 --notify-cmd 提醒人。

只走 Streamable HTTP：Latent 要以常驻服务端形态运行（`mcp_server.py --corpus … --http 127.0.0.1:8765
--token <普通>`），这样落档写进去，聊天那边的检索和自动浮现立刻看得到。

环境变量（命令行参数优先）：
  LATENT_MCP_URL          必填，例如 http://127.0.0.1:8765/mcp
  LATENT_MCP_TOKEN        服务端 --token
  LATENT_ARCHIVE_SOURCE   对话来源，形如 claude-code:<文件或目录> 或 jsonl:<文件>
  LATENT_ARCHIVE_ME       记忆主人（AI 这一方）的名字，正文里的「我」；默认「我」
  LATENT_ARCHIVE_USER     对方的名字；默认「对方」
  LATENT_ARCHIVE_MODEL_CMD  一次性模型命令，prompt 从 stdin 进、正文从 stdout 出
  LATENT_ARCHIVE_STATE_DIR  游标账本放在哪，默认 ~/.cache/latent-archive

同一时间只跑一个（锁文件）；聊天进程和落档进程互不等待。
"""

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from time_context import TimeContext  # noqa: E402

DEFAULT_MODEL_CMD = 'claude -p --tools "" --effort low'
WEEKDAYS = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
LOCK_STALE_SECONDS = 2 * 3600       # 锁文件超过这么久还在，当成上次进程死了留下的
FACT_TAGS, FACT_KINDS = {"life", "work", "meta"}, {"event", "state"}

# 宿主注入进对话的东西不是人说的话，落档前剔掉，免得把旧记忆当新事再写一遍
_STRIP_RES = [
    re.compile(r"<system-reminder>[\s\S]*?</system-reminder>"),
    re.compile(r"<passive_recall_context\b[\s\S]*?</passive_recall_context>"),
    re.compile(r"以下是 Latent 记忆库针对这条消息自动浮现的历史资料[\s\S]*?〔历史证据结束〕"),
]
# 以这些标签开头的 user 消息是宿主自己的事件（后台任务完成通知、斜杠命令回显），整条不算
_HOST_EVENT_RE = re.compile(r"^<(task-notification|command-name|command-message|command-args|"
                            r"local-command-stdout|local-command-stderr|local-command-caveat|"
                            r"bash-input|bash-stdout|bash-stderr)>")


@dataclass
class Msg:
    key: str
    epoch: float
    role: str          # "user" / "assistant"
    text: str


# ---------------------------------------------------------------- 读来源

def _epoch(value):
    if isinstance(value, (int, float)):
        return float(value) / (1000 if value > 1e11 else 1)
    dt = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _clean(text):
    for pattern in _STRIP_RES:
        text = pattern.sub("", text)
    return text.strip()


def _claude_code_text(message):
    """Claude Code 一行的 message.content → 只取说出来的文字；thinking、工具调用和工具结果都不算。"""
    content = (message or {}).get("content")
    if isinstance(content, str):
        return content
    return "\n".join(block.get("text", "") for block in content or []
                     if isinstance(block, dict) and block.get("type") == "text")


def read_claude_code(path):
    files = sorted(Path(path).glob("*.jsonl")) if Path(path).is_dir() else [Path(path)]
    out = []
    for file in files:
        for line in file.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("isSidechain") or row.get("isMeta"):
                continue
            att = row.get("attachment") or {}
            if row.get("type") == "attachment" and att.get("type") == "queued_command" \
                    and (att.get("origin") or {}).get("kind") == "human":
                # 模型回复途中人插进来的话，记成 attachment 而不是 user 行（Claude Code 2026.10 实测）
                role, text = "user", _clean(str(att.get("prompt") or ""))
            elif row.get("type") in {"user", "assistant"}:
                role, text = row["type"], _clean(_claude_code_text(row.get("message")))
            else:
                continue
            if not text or not row.get("timestamp") or (role == "user" and _HOST_EVENT_RE.match(text)):
                continue
            out.append(Msg(row.get("uuid") or _h(line), _epoch(row["timestamp"]), role, text))
    return out


def read_jsonl(path):
    """通用格式：每行 {"ts": ISO 或 epoch, "role": "user"|"assistant", "text": "…", "id": 可选}。"""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        text = _clean(str(row.get("text") or ""))
        if text and row.get("role") in {"user", "assistant"}:
            out.append(Msg(str(row.get("id") or _h(line)), _epoch(row["ts"]), row["role"], text))
    return out


READERS = {"claude-code": read_claude_code, "jsonl": read_jsonl}


def load_messages(source):
    kind, _, path = source.partition(":")
    if kind not in READERS or not path:
        raise ValueError(f"认不出对话来源：{source!r}（要写成 claude-code:<路径> 或 jsonl:<路径>）")
    seen, out = set(), []
    for msg in sorted(READERS[kind](path), key=lambda m: m.epoch):
        if msg.key not in seen:          # 恢复会话会把旧消息复制进新文件，按 key 去重
            seen.add(msg.key)
            out.append(msg)
    return out


# ---------------------------------------------------------------- 账本与锁

def _h(*parts):
    return hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).hexdigest()[:16]


class Ledger:
    """一个来源一份：游标（时间＋同一时刻已处理的 key）、写到一半的批次、失败计数、最近记录。"""

    def __init__(self, state_dir, source):
        self.path = Path(state_dir).expanduser() / f"{_h('source', source)}.json"
        try:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.data = {}
        for key, empty in (("cursor_epoch", 0.0), ("cursor_keys", []), ("inflight", None),
                           ("passed_key", None), ("streak", 0), ("history", [])):
            self.data.setdefault(key, empty)

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    def after_cursor(self, msgs):
        at, keys = self.data["cursor_epoch"], set(self.data["cursor_keys"])
        return [m for m in msgs if m.epoch > at or (m.epoch == at and m.key not in keys)]

    def advance(self, batch_msgs):
        last = batch_msgs[-1].epoch
        keys = [m.key for m in batch_msgs if m.epoch == last]
        if last == self.data["cursor_epoch"]:
            keys = sorted(set(keys) | set(self.data["cursor_keys"]))
        self.data.update(cursor_epoch=last, cursor_keys=keys, inflight=None, passed_key=None)


class Lock:
    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if time.time() - self.path.stat().st_mtime > LOCK_STALE_SECONDS:
                self.path.unlink()
        except OSError:
            pass
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            self.held = False
            return self
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        self.held = True
        return self

    def __exit__(self, *exc):
        if self.held:
            self.path.unlink(missing_ok=True)


# ---------------------------------------------------------------- 切批

@dataclass
class Batch:
    day: str
    msgs: list
    closed: bool       # 这一天已经过去：最后一次机会
    capped: bool       # 当天待整理的超过 --max-chars，先切最老的一段


def plan_batch(pending, tc, now, cfg, flush=False):
    if not pending:
        return None
    day = tc.local_date(pending[0].epoch)
    same_day = [m for m in pending if tc.local_date(m.epoch) == day]
    msgs, size = [], 0
    for m in same_day:
        if msgs and size + len(m.text) > cfg.max_chars:
            break
        msgs.append(m)
        size += len(m.text)
    batch = Batch(day, msgs, closed=day < tc.local_date(now), capped=len(msgs) < len(same_day))
    if flush:
        return batch
    stale = now - pending[0].epoch >= cfg.stale_minutes * 60
    if not (batch.closed or batch.capped or len(same_day) >= cfg.min_messages or stale):
        return None
    if {m.role for m in msgs} != {"user", "assistant"} and not batch.closed:
        return None        # 只有一方在说话：等对话真正发生
    return batch


def render_transcript(msgs, tc, me, user):
    names = {"user": user, "assistant": me}
    return "\n".join(f"[{datetime.fromtimestamp(m.epoch, tc._zone):%H:%M}] {names[m.role]}：{m.text}"
                     for m in msgs)


# ---------------------------------------------------------------- 模型与形状闸

NO_TOOLS = ("你没有任何工具，素材已全部在下面，不用也不能去查文件或仓库。"
            "要拒绝或觉得不值得写，就只输出 PASS 四个字母——这是正常出口；"
            "任何开场白、解释或工具调用都会被当成正文存进记忆库，所以一个字都不要多写。")


def head_line(day):
    return f"# 对话整理 · {day}"


def draft_prompt(batch, transcript, me, user, retry=""):
    weekday = WEEKDAYS[datetime.strptime(batch.day, "%Y-%m-%d").weekday()]
    return "\n".join([
        "<archive_task>",
        f"[后台落档 · 初稿] 这是记忆库的后台整理任务，不是{user}发来的消息。下面是 {batch.day}（{weekday}）"
        f"{me}和{user}之间的 {len(batch.msgs)} 条真实对话，方括号是本地时间。",
        "判断这段有没有值得长期记住的内容；没有就只输出 PASS。有就输出 Markdown：",
        f"- 首行一字不差写「{head_line(batch.day)}」，输出的第一个字符就是这个「#」。",
        f"- 正文用{me}的第一人称：「我」指{me}，{user}直接写名字。按主题归并，不按聊天轮次写流水账；"
        "写具体发生了什么、说过的原话、定下的约定，不写评语。同一件事只写一次，全文不超过 1500 字。",
        "- 时段按方括号里的时间判断，00:00—05:59 写凌晨；对话里提到的旧事只能写成回忆，不能写成这一天发生。",
        "- 只写对话里有的，不补、不猜。",
        "- 最后一行写「当下状态：……」，一句话说清这段聊到的事现在是什么状态（约定成立／还在处理／已解决……）。",
        NO_TOOLS, *([retry] if retry else []),
        "<conversation>", transcript, "</conversation>", "</archive_task>"])


def review_prompt(batch, transcript, draft, me, user, retry=""):
    return "\n".join([
        "<archive_task>",
        f"[后台落档 · 自审] 下面的初稿还没落盘。对照原始对话逐项核对，直接输出修订后的完整稿：",
        f"1. 谁做的、谁说的：{me}和{user}不能写反。",
        f"2. 日期与时段：这一天是 {batch.day}；00:00—05:59 是凌晨；旧事只能写成回忆。",
        f"3. 视角：正文是{me}的第一人称，「我」指{me}。",
        "4. 没有编造：对话里找不到出处的句子删掉。",
        "5. 去重：同一件事只留一次，删掉按轮次复述的流水账，全文不超过 1500 字。",
        f"修订稿首行仍是「{head_line(batch.day)}」，最后一行仍是「当下状态：……」。"
        "整篇不值得保留就只输出 PASS。不要解释你检查了什么。",
        NO_TOOLS, *([retry] if retry else []),
        "<conversation>", transcript, "</conversation>",
        "<draft>", draft, "</draft>", "</archive_task>"])


def facts_prompt(batch, transcript, body, me, user):
    return "\n".join([
        "<archive_task>",
        f"[后台落档 · 事实拆条] 把下面这段对话里值得单独记住的小事拆出来，一件事一句，8–120 个字，写清主语"
        f"（「我」指{me}，{user}直接写名字）。优先：买了什么、定了什么约定、身体和情绪、一句原话、一个具体数字。",
        '每条单独一行 JSON：{"fact":"……","tag":"life","kind":"event"}。tag 取 life／work／meta，'
        "kind 取 event（发生了一件事）或 state（一段时间里的状态）。没有就只输出 PASS。",
        NO_TOOLS, "<conversation>", transcript, "</conversation>",
        "<summary>", body, "</summary>", "</archive_task>"])


def _argv(cmd):
    return shlex.split(cmd, posix=os.name != "nt") if isinstance(cmd, str) else list(cmd)


def run_model(cmd, prompt, timeout):
    argv = _argv(cmd)
    proc = subprocess.run(argv, input=prompt.encode("utf-8"), capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"模型命令退出码 {proc.returncode}：{proc.stderr.decode('utf-8', 'replace')[-300:]}")
    return proc.stdout.decode("utf-8", "replace")


def clean_reply(text):
    """None＝PASS；否则去掉代码围栏后的正文。"""
    raw = str(text or "").strip()
    fenced = re.match(r"^```(?:markdown|md)?\s*\n([\s\S]*?)\n?```$", raw, re.IGNORECASE)
    raw = (fenced.group(1) if fenced else raw).strip()
    return None if raw.upper().rstrip(".。") == "PASS" else raw


_STATE_RE = re.compile(r"^\s*当下状态[：:]\s*(.+?)\s*$", re.MULTILINE)


def is_shaped(text, day):
    return text.splitlines()[0].strip() == head_line(day) and bool(_STATE_RE.search(text))


def retry_note(day, bad):
    return (f"⚠ 上一次的输出没有通过落盘闸（开头是「{bad}」），这是最后一次机会：从「{head_line(day)}」"
            "一字不差地写起，最后一行写「当下状态：……」。不值得写就只输出 PASS。")


def generate_shaped(cmd, build, day, timeout):
    """生成一次；形状不对就带着纠正提示再生成一次（只一次：管道级故障掷第三次也没用）。"""
    first = clean_reply(run_model(cmd, build(""), timeout))
    if first is None or is_shaped(first, day):
        return first
    bad = re.sub(r"\s+", " ", first)[:60]
    return clean_reply(run_model(cmd, build(retry_note(day, bad)), timeout))


def split_state(text, day, today):
    """约定首行和「当下状态」从正文里拿掉：首行只是形状闸的锚，留着会在记忆库文件里多一个一级标题；
    current_state 是 latent_append 的必填参数，服务端会自己写回正文末尾。
    记录的日期标签由服务端按写入当天打；补写前一天的批次时，正文第一句写明是哪天的事。"""
    found = list(_STATE_RE.finditer(text))[-1]
    body = (text[:found.start()] + text[found.end():]).strip()
    body = body.split("\n", 1)[1].strip() if "\n" in body else ""
    if day != today:
        body = f"（这是 {day} 的对话，{today} 补记。）\n\n{body}"
    return body, found.group(1)


def parse_facts(text, day):
    out = []
    for line in (clean_reply(text) or "").splitlines():
        try:
            item = json.loads(line.strip())
        except ValueError:
            continue
        fact = str((item or {}).get("fact", "")).strip() if isinstance(item, dict) else ""
        tag, kind = item.get("tag", "life") if fact else "", item.get("kind", "event") if fact else ""
        if 8 <= len(fact) <= 120 and tag in FACT_TAGS and kind in FACT_KINDS:
            out.append({"fact": fact, "tag": tag, "kind": kind, "event_date": day})
    return out


# ---------------------------------------------------------------- 写进 Latent

def call_latent(url, token, tool, arguments, timeout=60):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                       "params": {"name": tool, "arguments": arguments}}, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(urllib.request.Request(url, data=body, headers=headers), timeout=timeout) as resp:
        raw = resp.read().decode("utf-8").strip()
    if raw.startswith(("event:", "data:")):
        raw = "\n".join(line[5:].strip() for line in raw.splitlines() if line.startswith("data:"))
    msg = json.loads(raw)
    if msg.get("error"):
        raise RuntimeError(f"{tool} 协议错误：{msg['error'].get('message')}")
    result = msg.get("result") or {}
    text = "\n".join(c.get("text", "") for c in result.get("content") or [] if c.get("type") == "text")
    if result.get("isError"):
        raise RuntimeError(f"{tool} 被拒：{text[:300]}")
    return text


def _probe(body):
    """正文里第一句够长的话，用来对账「上次是不是其实已经写进去了」。"""
    for line in body.splitlines():
        line = line.strip().lstrip("#-* ").strip()
        if len(line) >= 12 and not line.startswith("（这是 "):
            return line[:40]
    return ""


def already_written(cfg, inflight):
    probe = _probe(inflight["text"])
    if not probe:
        return False
    found = call_latent(cfg.url, cfg.token, "latent_search", {"query": probe, "topN": 5})
    return probe in found


def append(cfg, inflight):
    args = {"text": inflight["text"], "current_state": inflight["current_state"]}
    if inflight.get("facts"):
        args["facts"] = inflight["facts"]
    text = call_latent(cfg.url, cfg.token, "latent_append", args)
    found = re.search(r"recordId=([0-9a-f]{16})", text)
    return found.group(1) if found else ""


# ---------------------------------------------------------------- 一次检查

def _fail(ledger, cfg, message):
    ledger.data["streak"] += 1
    ledger.data["last_error"] = {"at": time.time(), "message": message[:500]}
    ledger.save()
    print(f"[background_archive] 失败（连续第 {ledger.data['streak']} 次）：{message}", file=sys.stderr)
    if cfg.notify_cmd and ledger.data["streak"] % cfg.notify_after == 0:
        try:
            subprocess.run(_argv(cfg.notify_cmd) + [f"Latent 后台落档连续失败 {ledger.data['streak']} 次：{message[:120]}"],
                           timeout=30)
        except Exception as e:      # 提醒失败不能盖掉原来的失败
            print(f"[background_archive] 提醒命令失败：{e}", file=sys.stderr)
    return {"status": "failed", "error": message, "streak": ledger.data["streak"]}


def _done(ledger, batch_msgs, day, record_id):
    ledger.advance(batch_msgs)
    ledger.data["streak"] = 0
    ledger.data["last_ok"] = time.time()
    ledger.data["history"] = (ledger.data["history"] + [
        {"day": day, "recordId": record_id, "messages": len(batch_msgs), "at": time.time()}])[-20:]
    ledger.save()
    return {"status": "written", "day": day, "recordId": record_id, "messages": len(batch_msgs)}


def run_once(cfg, now=None, flush=False):
    now = time.time() if now is None else now
    tc = cfg.tc
    ledger = Ledger(cfg.state_dir, cfg.source)
    with Lock(ledger.path.with_suffix(".lock")) as lock:
        if not lock.held:
            return {"status": "locked"}
        msgs = load_messages(cfg.source)
        inflight = ledger.data["inflight"]
        if inflight:
            # 上次写到一半进程没了：拿存下的同一份正文对账，绝不重新生成
            batch_msgs = [m for m in msgs if m.key in set(inflight["keys"])]
            if not batch_msgs:
                return _fail(ledger, cfg, "上次未完成的批次在来源里找不到了（记录被删或换了位置），"
                                          "账本里 inflight 留着原稿，人工核对后删掉 inflight 再跑")
            try:
                if already_written(cfg, inflight):
                    return dict(_done(ledger, batch_msgs, inflight["day"], ""), recovered="already_written")
                return dict(_done(ledger, batch_msgs, inflight["day"], append(cfg, inflight)), recovered="reappended")
            except Exception as e:
                return _fail(ledger, cfg, f"对账上次未完成的批次失败：{e}")
        batch = plan_batch(ledger.after_cursor(msgs), tc, now, cfg, flush)
        if batch is None:
            return {"status": "idle"}
        final = batch.closed or batch.capped or flush     # 这一段不会再长，或者调用方要求现在就收
        if batch.msgs[-1].key == ledger.data["passed_key"] and not final:
            return {"status": "idle", "reason": "已判过不值得写，等新消息"}
        transcript = render_transcript(batch.msgs, tc, cfg.me, cfg.user)
        try:
            text = generate_shaped(cfg.model_cmd, lambda r: draft_prompt(batch, transcript, cfg.me, cfg.user, r),
                                   batch.day, cfg.timeout)
            if text is not None and cfg.review and is_shaped(text, batch.day):
                draft = text
                text = generate_shaped(cfg.model_cmd,
                                       lambda r: review_prompt(batch, transcript, draft, cfg.me, cfg.user, r),
                                       batch.day, cfg.timeout)
        except Exception as e:
            return _fail(ledger, cfg, f"模型调用失败：{e}")
        if text is None:
            # PASS 是判断不是故障。这一段不会再长（那天已过去／满批截断／flush）：越过去；否则等新消息
            if final:
                ledger.advance(batch.msgs)
            else:
                ledger.data["passed_key"] = batch.msgs[-1].key
            ledger.data["streak"] = 0
            ledger.save()
            return {"status": "pass", "day": batch.day, "advanced": final}
        if not is_shaped(text, batch.day):
            return _fail(ledger, cfg, f"输出不合约定形状，未落盘：{re.sub(chr(10), ' ', text)[:120]}")
        body, state = split_state(text, batch.day, tc.local_date(now))
        facts = []
        if cfg.facts:
            try:
                facts = parse_facts(run_model(cfg.model_cmd, facts_prompt(batch, transcript, body, cfg.me, cfg.user),
                                              cfg.timeout), batch.day)
            except Exception as e:      # 事实失败不拖垮正文
                print(f"[background_archive] 事实拆条失败，这批只写正文：{e}", file=sys.stderr)
        ledger.data["inflight"] = {"day": batch.day, "text": body, "current_state": state, "facts": facts,
                                   "keys": [m.key for m in batch.msgs]}
        ledger.save()          # 先记下要写什么，再写：中途死掉下次按这份对账
        try:
            record_id = append(cfg, ledger.data["inflight"])
        except Exception as e:
            return _fail(ledger, cfg, f"写入 Latent 失败：{e}")
        return _done(ledger, batch.msgs, batch.day, record_id)


def flush(cfg, now=None, max_batches=50):
    """不看门槛，把游标之后的全部整理掉；一批一批来，直到没有待整理的。任何一批失败就停，返回那次的结果。"""
    results = []
    for _ in range(max_batches):
        result = run_once(cfg, now, flush=True)
        if result["status"] in {"idle", "locked", "failed"}:
            return {"status": "flushed" if result["status"] == "idle" else result["status"],
                    "batches": results, **({"error": result["error"]} if "error" in result else {})}
        results.append(result)
    return {"status": "failed", "error": f"连续 {max_batches} 批还没整理完", "batches": results}


# ---------------------------------------------------------------- 命令行

def build_config(argv):
    env = os.environ.get
    p = argparse.ArgumentParser(description="Latent 后台落档")
    p.add_argument("command", choices=["run", "watch", "status", "flush"])
    p.add_argument("--url", default=env("LATENT_MCP_URL"))
    p.add_argument("--token", default=env("LATENT_MCP_TOKEN"))
    p.add_argument("--source", default=env("LATENT_ARCHIVE_SOURCE"))
    p.add_argument("--me", default=env("LATENT_ARCHIVE_ME", "我"))
    p.add_argument("--user", default=env("LATENT_ARCHIVE_USER", "对方"))
    p.add_argument("--model-cmd", default=env("LATENT_ARCHIVE_MODEL_CMD", DEFAULT_MODEL_CMD))
    p.add_argument("--state-dir", default=env("LATENT_ARCHIVE_STATE_DIR", "~/.cache/latent-archive"))
    p.add_argument("--timezone", default=None, help="IANA 名或 UTC+08:00；不填用东八区")
    p.add_argument("--min-messages", type=int, default=60)
    p.add_argument("--stale-minutes", type=int, default=90)
    p.add_argument("--max-chars", type=int, default=30000)
    p.add_argument("--no-review", dest="review", action="store_false")
    p.add_argument("--facts", action="store_true", help="同时拆事实条目（部署开了自动浮现事实模式才有用）")
    p.add_argument("--timeout", type=int, default=600, help="单次模型调用超时秒数")
    p.add_argument("--interval", type=int, default=600, help="watch 模式的检查间隔秒数")
    p.add_argument("--notify-cmd", default=env("LATENT_ARCHIVE_NOTIFY_CMD"),
                   help="连续失败时执行，最后一个参数是提醒文字")
    p.add_argument("--notify-after", type=int, default=3)
    cfg = p.parse_args(argv)
    cfg.tc = TimeContext(cfg.timezone) if cfg.timezone else TimeContext.default()
    if cfg.command != "status" and not (cfg.url and cfg.source):
        p.error("要配 LATENT_MCP_URL 和 LATENT_ARCHIVE_SOURCE（或 --url／--source）")
    return cfg


def main(argv):
    if argv[1:2] == ["--selftest"]:
        return _selftest()
    cfg = build_config(argv[1:])
    if cfg.command == "status":
        if not cfg.source:
            print("要配 LATENT_ARCHIVE_SOURCE 或 --source 才知道看哪本账", file=sys.stderr)
            return 2
        data = Ledger(cfg.state_dir, cfg.source).data
        data["cursor"] = cfg.tc.isoformat(data["cursor_epoch"]) if data["cursor_epoch"] else None
        print(json.dumps(data, ensure_ascii=False, indent=1))
        return 0
    if cfg.command == "flush":
        result = flush(cfg)
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["status"] == "flushed" else 1
    while True:
        try:
            result = run_once(cfg)
        except Exception as e:      # 来源读不了这类：记一笔，常驻模式下一轮再试
            result = {"status": "failed", "error": str(e)}
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if cfg.command == "run":
            return 1 if result["status"] == "failed" else 0
        time.sleep(cfg.interval)


# ---------------------------------------------------------------- 自检（合成夹具，不读真实库、不调真模型）

FAKE_MODEL = r'''
import json, os, sys
prompt = sys.stdin.read()
mode = os.environ.get("FAKE_MODE", "ok")
log = os.environ.get("FAKE_LOG")
if log:
    open(log, "a", encoding="utf-8").write(prompt.split("\n")[1][:20] + "\n")
day = prompt.split("这一天是 ")[1][:10] if "[后台落档 · 自审]" in prompt else prompt.split("下面是 ")[1][:10] if "[后台落档 · 初稿]" in prompt else ""
if mode == "pass":
    print("PASS")
elif mode == "refuse":
    print("我不能处理这段私人对话。")
elif mode == "fixonretry" and "[后台落档 · 初稿]" in prompt and "最后一次机会" not in prompt:
    print("好的，我先看一下仓库。")
elif "事实拆条" in prompt:
    print(json.dumps({"fact": "林夏在楼下面馆点了一碗牛肉面", "tag": "life", "kind": "event"}, ensure_ascii=False))
    print("这一行不是 JSON")
    print(json.dumps({"fact": "短", "tag": "life"}, ensure_ascii=False))
else:
    print("```markdown\n# 对话整理 · " + day + "\n\n晚上林夏说楼下新开的面馆汤头很浓，我们约好周六一起去吃牛肉面。\n\n当下状态：周六吃面的约定成立\n```")
'''


def _selftest():
    import tempfile
    import threading
    from types import SimpleNamespace
    from mcp_server import MemoryServer, make_corpus_loader, make_http_server
    from session_thread import ThreadStore

    tc = TimeContext("UTC+08:00")
    day0 = tc.midnight_epoch("2026-09-01")

    # 1. Claude Code jsonl：只取说出来的文字，宿主注入和事件都剔掉
    with tempfile.TemporaryDirectory() as tmp:
        rows = [
            {"type": "user", "uuid": "u1", "timestamp": "2026-09-01T12:00:00Z",
             "message": {"content": "<system-reminder>别管</system-reminder>晚上吃什么"}},
            {"type": "assistant", "uuid": "a1", "timestamp": "2026-09-01T12:00:05Z",
             "message": {"content": [{"type": "thinking", "thinking": "想一想"},
                                     {"type": "text", "text": "楼下面馆"},
                                     {"type": "tool_use", "name": "Bash", "input": {}}]}},
            {"type": "user", "uuid": "u2", "timestamp": "2026-09-01T12:00:06Z",
             "message": {"content": [{"type": "tool_result", "content": "ok"}]}},
            {"type": "user", "uuid": "u3", "timestamp": "2026-09-01T12:01:00Z",
             "message": {"content": "<task-notification>\n<status>completed</status>"}},
            {"type": "user", "uuid": "u4", "timestamp": "2026-09-01T12:02:00Z", "isSidechain": True,
             "message": {"content": "子代理的话"}},
            {"type": "summary", "summary": "压缩摘要"},
            {"type": "attachment", "uuid": "q1", "timestamp": "2026-09-01T12:00:03Z",
             "attachment": {"type": "queued_command", "prompt": "要辣的", "origin": {"kind": "human"}}},
            {"type": "attachment", "uuid": "q2", "timestamp": "2026-09-01T12:00:04Z",
             "attachment": {"type": "hook_additional_context", "content": ["注入的旧记忆"]}},
        ]
        path = Path(tmp) / "s.jsonl"
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n不是json\n", encoding="utf-8")
        (Path(tmp) / "s2.jsonl").write_text(json.dumps(rows[0], ensure_ascii=False), encoding="utf-8")
        got = load_messages(f"claude-code:{tmp}")
        assert [(m.role, m.text) for m in got] == [("user", "晚上吃什么"), ("user", "要辣的"),
                                                   ("assistant", "楼下面馆")], got
        try:
            load_messages("whatever:/x")
            raise AssertionError("认不出的来源要报错")
        except ValueError:
            pass

    # 2. 切批门槛
    cfg = SimpleNamespace(max_chars=30000, min_messages=4, stale_minutes=90)

    def mk(n, start, step=60):
        return [Msg(f"k{start}-{i}", start + i * step, "user" if i % 2 == 0 else "assistant", f"第{i}句话")
                for i in range(n)]
    now = day0 + 12 * 3600
    assert plan_batch(mk(2, now - 600), tc, now, cfg) is None, "条数不够、也没等太久：不动"
    assert plan_batch(mk(4, now - 600), tc, now, cfg).msgs, "条数够了：动"
    assert plan_batch(mk(2, now - 100 * 60), tc, now, cfg).msgs, "最老一条等了 90 分钟以上：动"
    assert plan_batch(mk(2, now - 100 * 60)[:1], tc, now, cfg) is None, "只有一方在说话：不动"
    old = mk(1, day0 - 3600)
    batch = plan_batch(old + mk(3, now - 600), tc, now, cfg)
    assert batch.day == "2026-08-31" and batch.closed and len(batch.msgs) == 1, "昨天剩的尾巴单独收，哪怕只一条"
    capped = plan_batch(mk(6, now - 600), tc, now, SimpleNamespace(max_chars=8, min_messages=99, stale_minutes=999))
    assert capped.capped and len(capped.msgs) == 2, "超过字数上限：先切最老的一段"

    # 3. 形状闸
    assert clean_reply("```md\n# 对话整理 · 2026-09-01\nx\n```") == "# 对话整理 · 2026-09-01\nx"
    assert clean_reply(" PASS ") is None
    assert is_shaped("# 对话整理 · 2026-09-01\n正文\n当下状态：成立", "2026-09-01")
    assert not is_shaped("好的，我来整理\n# 对话整理 · 2026-09-01\n当下状态：成立", "2026-09-01")
    assert not is_shaped("# 对话整理 · 2026-09-01\n没有状态行", "2026-09-01")
    assert split_state("# 对话整理 · d\n正文\n当下状态：成立", "d", "d") == ("正文", "成立")
    assert split_state("# 对话整理 · d\n正文\n当下状态：成立", "d", "e")[0].startswith("（这是 d 的对话，e 补记。）")
    facts = parse_facts('{"fact":"林夏在楼下面馆点了一碗牛肉面","tag":"life"}\n坏行\n{"fact":"短"}\n'
                        '{"fact":"林夏说汤头很浓很好喝","tag":"bad"}', "2026-09-01")
    assert facts == [{"fact": "林夏在楼下面馆点了一碗牛肉面", "tag": "life", "kind": "event",
                      "event_date": "2026-09-01"}], facts

    # 4. 端到端：真服务端、假模型
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        corpus, index_dir = root / "timeline", root / "index"
        corpus.mkdir()
        index_dir.mkdir()
        source_dirs, loader = make_corpus_loader(corpus, index_dir)
        srv = make_http_server(MemoryServer(index=loader(), thread_store=ThreadStore(), corpus_dir=corpus,
                                            index_dir=index_dir, source_dirs=source_dirs, loader=loader,
                                            time_context=tc), port=0, token="plain")
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        fake = root / "fake_model.py"
        fake.write_text(FAKE_MODEL, encoding="utf-8")
        chat = root / "chat.jsonl"
        lines = [{"ts": now - 600 + i * 60, "role": "user" if i % 2 == 0 else "assistant",
                  "text": ["楼下新开了面馆", "汤头怎么样", "很浓，周六去吃吧", "好，周六一起去"][i]} for i in range(4)]
        chat.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in lines), encoding="utf-8")
        notes = root / "notify.txt"
        notifier = root / "notify.py"
        notifier.write_text(f"import sys\nopen({str(notes)!r}, 'a', encoding='utf-8').write(sys.argv[-1] + '\\n')\n",
                            encoding="utf-8")
        log = root / "calls.txt"
        os.environ["FAKE_LOG"] = str(log)
        run_cfg = SimpleNamespace(
            url=f"http://127.0.0.1:{srv.server_address[1]}/mcp", token="plain", source=f"jsonl:{chat}",
            me="我", user="林夏", model_cmd=[sys.executable, str(fake)], state_dir=str(root / "state"),
            tc=tc, min_messages=4, stale_minutes=90, max_chars=30000, review=True, facts=True, timeout=30,
            notify_cmd=f"{shlex.quote(sys.executable)} {shlex.quote(str(notifier))}", notify_after=3)

        def calls():
            return log.read_text(encoding="utf-8").splitlines() if log.exists() else []

        def files():
            return "".join(p.read_text(encoding="utf-8") for p in corpus.rglob("*.md"))

        # 4a. 模型拒绝（自由文本）：带纠正提示重来一次，还不合形状就不落盘、游标不动、计失败；第 3 次提醒
        os.environ["FAKE_MODE"] = "refuse"
        for i in range(3):
            r = run_once(run_cfg, now)
            assert r["status"] == "failed" and r["streak"] == i + 1, r
        assert len(calls()) == 6, "每次检查只重试一次"
        assert "牛肉面" not in files() and Ledger(run_cfg.state_dir, run_cfg.source).data["cursor_epoch"] == 0
        assert notes.read_text(encoding="utf-8").count("连续失败 3 次") == 1, "连续失败到 3 次要提醒一次"
        # 4b. PASS：游标不动、记下判过的位置，同一段不再反复问模型
        os.environ["FAKE_MODE"] = "pass"
        assert run_once(run_cfg, now)["status"] == "pass"
        n = len(calls())
        assert run_once(run_cfg, now)["status"] == "idle" and len(calls()) == n, "判过 PASS 的同一段不再调模型"
        # 4c. 新消息来了、第一次跑偏第二次改对：初稿重试一次 → 自审 → 事实 → 写进去，游标推进
        lines.append({"ts": now - 30, "role": "user", "text": "记得带伞"})
        chat.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in lines), encoding="utf-8")
        os.environ["FAKE_MODE"] = "fixonretry"
        log.unlink()
        r = run_once(run_cfg, now)
        assert r["status"] == "written" and re.fullmatch(r"[0-9a-f]{16}", r["recordId"]), r
        assert [c.split("]")[0] for c in calls()] == ["[后台落档 · 初稿", "[后台落档 · 初稿", "[后台落档 · 自审",
                                                      "[后台落档 · 事实拆条"], calls()
        text = files()
        assert "周六一起去吃牛肉面" in text and "当下状态：周六吃面的约定成立" in text, text
        ledger = Ledger(run_cfg.state_dir, run_cfg.source)
        assert ledger.data["streak"] == 0 and ledger.data["inflight"] is None
        assert ledger.data["cursor_epoch"] == lines[-1]["ts"]
        fact_rows = [json.loads(line) for p in (root / "事实库").rglob("*.jsonl")
                     for line in p.read_text(encoding="utf-8").splitlines()]
        assert [f["fact"] for f in fact_rows] == ["林夏在楼下面馆点了一碗牛肉面"], fact_rows
        # 4d. 没有新消息：不调模型
        n = len(calls())
        assert run_once(run_cfg, now + 3600)["status"] == "idle" and len(calls()) == n
        # 4e. 写到一半进程死了（服务端已写进、账本没来得及推游标）：对账发现已写，不重复写
        lines += [{"ts": now + 60 + i, "role": "user" if i % 2 == 0 else "assistant", "text": f"第二段第{i}句"}
                  for i in range(4)]
        chat.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in lines), encoding="utf-8")
        body = "下午林夏说要把阳台的薄荷换一个大一点的花盆再种一次。"
        inflight = {"day": "2026-09-01", "text": body, "current_state": "还没换盆", "facts": [],
                    "keys": [str(_h(json.dumps(r, ensure_ascii=False))) for r in lines[-4:]]}
        append(run_cfg, inflight)
        ledger.data["inflight"] = inflight
        ledger.save()
        r = run_once(run_cfg, now + 3600)
        assert r.get("recovered") == "already_written", r
        assert files().count("换一个大一点的花盆") == 1, "已经写进去的不能再写一遍"
        assert Ledger(run_cfg.state_dir, run_cfg.source).data["cursor_epoch"] == lines[-1]["ts"]
        # 4f. 对账没找到：用存下的同一份正文补写（不重新生成）
        n = len(calls())
        ledger = Ledger(run_cfg.state_dir, run_cfg.source)
        ledger.data["inflight"] = dict(inflight, text="傍晚林夏把自行车的链条上了油，说骑起来安静多了。")
        ledger.save()
        r = run_once(run_cfg, now + 3600)
        assert r.get("recovered") == "reappended" and "链条上了油" in files() and len(calls()) == n, r
        # 4g. 锁：已经有一个在跑，第二个直接退
        lock_path = Ledger(run_cfg.state_dir, run_cfg.source).path.with_suffix(".lock")
        lock_path.write_text("1")
        assert run_once(run_cfg, now + 7200)["status"] == "locked"
        lock_path.unlink()
        # 4h. 服务端连不上：计失败，游标不动
        lines.append({"ts": now + 7000, "role": "assistant", "text": "晚安"})
        lines.append({"ts": now + 7001, "role": "user", "text": "晚安"})
        chat.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in lines), encoding="utf-8")
        os.environ["FAKE_MODE"] = "ok"
        before = Ledger(run_cfg.state_dir, run_cfg.source).data["cursor_epoch"]
        r = run_once(SimpleNamespace(**dict(vars(run_cfg), url="http://127.0.0.1:9/mcp")), now + 7001 + 100 * 60)
        assert r["status"] == "failed" and Ledger(run_cfg.state_dir, run_cfg.source).data["cursor_epoch"] == before, r
        # 4i. flush：不看门槛（刚来两条、没等够），全部收掉；模型 PASS 也越过去
        r = run_once(run_cfg, now + 7002)
        assert r.get("recovered") in {"reappended", "already_written"}, r   # 假模型每次写同一句，对账会认成已写
        lines += [{"ts": now + 7100, "role": "user", "text": "明早见"}, {"ts": now + 7101, "role": "assistant", "text": "明早见"}]
        chat.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in lines), encoding="utf-8")
        os.environ["FAKE_MODE"] = "pass"
        assert run_once(run_cfg, now + 7102)["status"] == "idle", "不到门槛 run 不动"
        r = flush(run_cfg, now + 7102)
        assert r["status"] == "flushed" and r["batches"][0]["advanced"], r
        assert Ledger(run_cfg.state_dir, run_cfg.source).data["cursor_epoch"] == lines[-1]["ts"]
        srv.shutdown()
        for key in ("FAKE_LOG", "FAKE_MODE"):
            os.environ.pop(key, None)
    print("selftest ok（background_archive：Claude Code 记录只取说出来的话、剔掉注入与事件 / 切批门槛与跨天收尾 / "
          "形状闸与一次纠正重试 / 拒绝不落盘、连续失败提醒 / PASS 不反复调模型 / 真服务端写入与事实 / "
          "中途死掉按原稿对账不重复 / 锁 / 服务端断连游标不动 / flush 不看门槛）")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
