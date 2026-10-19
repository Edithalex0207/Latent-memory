#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""原地续接参考实现：Claude Code 会话的上下文快满时，裁掉前面、原样留下最近一段，接着在“同一个窗口”里聊。

  python3 forge_session.py check <会话.jsonl>    # 看最近一轮的真实上下文多大、到没到门槛
  python3 forge_session.py forge <会话.jsonl>    # 到门槛就续接：写出新会话文件，打印新会话 id

不是 /compact：不跑摘要、不加“前情提要”。做法是把旧会话最近 --retain 那段 user/assistant 事件原样留下
（连 thinking 一起），换一个新的会话 id、重建父链，写成同目录下一个新的会话文件。之后宿主用
`claude --resume <新会话id>` 接着跑，模型读到的是一段完整的、自己的对话。

三个坑，都是真出过事的：
1. 父链必须重建。只改 sessionId 不动 parentUuid，恢复时沿父链往上走会撞到指向旧会话的 uuid，
   整段被当成断链丢掉。
2. thinking 必须保留。去掉之后模型会从历史里学到“这里不思考”，之后很多轮都压着不想。
3. 保留区间里没有一次完整的工具调用时，从更早的地方捞最近 1～2 次成功调用垫在最前面（去掉 thinking、
   工具结果压成短摘要）。否则模型看不到自己用过工具，会说自己不能用或者假装调用。

裁之前先保住被裁掉的那部分：
- `--archive`：先把游标之后没整理的对话按《后台落档》整理进记忆库（flush，不看门槛）；失败就不裁。
- 配了 LATENT_MCP_URL：再往 Latent 写一个续接检查点（`latent_thread_close`：最近的话头），
  下次换窗召回能接上；写不进去就不裁。

换进程不归这里管：这个脚本只写新文件、打印新 id。常驻进程型的宿主（`claude -p --input-format
stream-json`）起一个 `--resume <新id>` 的新进程，确认它起来了再关旧的；每轮起一次进程的宿主，
下一轮把 `--resume` 换成新 id 就行。旧会话文件不删。

环境变量：LATENT_MCP_URL、LATENT_MCP_TOKEN（同《后台落档》）；--archive 时还读 LATENT_ARCHIVE_* 那几个。
"""

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_THRESHOLD = 600_000    # 按 1M 上下文的模型定：留 40 万余量，一轮里的大工具输出、图片顶不到天花板
DEFAULT_RETAIN = 200_000       # 续接后留下的尾部；必须小于门槛，否则刚续完又越线
PRIMER_ROUNDS = 2
PRIMER_RESULT_CHARS = 200


# ---------------------------------------------------------------- 读会话

def load_events(path):
    """只要 user/assistant 事件；attachment、权限、文件快照这类 meta 行不带进新会话。"""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("type") in {"user", "assistant"}:
            out.append(event)
    return out


def context_tokens(usage):
    """一次调用的真实上下文 = input + cache_creation + cache_read。"""
    if not usage:
        return None
    return sum(int(usage.get(k) or 0) for k in
               ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))


def last_context_tokens(events):
    for event in reversed(events):
        if event.get("type") == "assistant":
            found = context_tokens((event.get("message") or {}).get("usage"))
            if found is not None:
                return found
    return None


# ---------------------------------------------------------------- 切点

def estimate(event):
    """粗估：JSON 长度 / 3。uuid、签名、转义都算进去了，绝对值偏大；有真实用量时只拿它分摊比例。"""
    return len(json.dumps(event, ensure_ascii=False)) // 3


def _is_real_user(event):
    """纯工具结果、meta 消息不算人说的话：从半截工具回合接起，模型会懵。"""
    if event.get("type") != "user" or event.get("isMeta"):
        return False
    content = (event.get("message") or {}).get("content")
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        return any(isinstance(b, dict) and b.get("type") != "tool_result" for b in content)
    return True


def find_boundary(events, retain, real_context=None):
    """从尾部往前累加到刚超过 retain 的位置，再往后找第一条人说的话作为干净边界。None＝不用裁。

    给了真实上下文就按“真实／估算”把每条等比缩放再累加：粗估偏大时，真实 18 万会被当成 44 万，
    裁一点点就付一次整段缓存重写。缩放顺带把系统提示、工具表这类不在记录里的固定开销摊进每条。"""
    if real_context and real_context <= retain:
        return None
    sizes = [estimate(e) for e in events]
    total = sum(sizes)
    scale = real_context / total if real_context and total else 1
    acc, cut = 0.0, 0
    for i in range(len(events) - 1, -1, -1):
        acc += sizes[i] * scale
        if acc > retain:
            cut = i + 1
            break
    if cut == 0:
        return None
    for i in range(cut, len(events)):
        if _is_real_user(events[i]):
            return i
    return None


# ---------------------------------------------------------------- 工具先例

def _blocks(event, kind):
    content = (event.get("message") or {}).get("content")
    return [b for b in content if isinstance(b, dict) and b.get("type") == kind] if isinstance(content, list) else []


def _tool_rounds(events):
    """完整且成功的工具回合 [(assistant 下标, user 下标)]，按时间顺序。"""
    rounds = []
    for i, event in enumerate(events):
        if event.get("type") != "assistant":
            continue
        for use in _blocks(event, "tool_use"):
            j = next((k for k in range(i + 1, len(events)) if events[k].get("type") == "user" and any(
                b.get("tool_use_id") == use.get("id") and not b.get("is_error")
                for b in _blocks(events[k], "tool_result"))), None)
            if j is not None:
                rounds.append((i, j))
                break
    return rounds


def _primer_assistant(event):
    message = dict(event["message"], content=[b for b in event["message"]["content"]
                                              if not (isinstance(b, dict) and b.get("type") == "thinking")])
    return dict(event, message=message)


def _primer_result(event):
    content = []
    for b in event["message"]["content"]:
        if isinstance(b, dict) and b.get("type") == "tool_result":
            body = b.get("content") if isinstance(b.get("content"), str) else json.dumps(b.get("content"), ensure_ascii=False)
            b = dict(b, content=f"成功：{(body or '')[:PRIMER_RESULT_CHARS]}")
        content.append(b)
    return dict(event, message=dict(event["message"], content=content))


def with_tool_primer(events, start):
    kept = events[start:]
    if _tool_rounds(kept):
        return kept
    rounds = _tool_rounds(events[:start])[-PRIMER_ROUNDS:]
    primer = []
    for a, u in rounds:               # 没有真实的成功回合就不垫，不编造
        primer += [_primer_assistant(events[a]), _primer_result(events[u])]
    return primer + kept


def rewrite_chain(events):
    """新会话 id；第一条 parentUuid=None，之后每条指向前一条。uuid 保留原值（落档按 uuid 去重靠它）。"""
    sid = str(uuid.uuid4())
    prev, out = None, []
    for event in events:
        current = event.get("uuid") or str(uuid.uuid4())
        out.append(dict(event, sessionId=sid, parentUuid=prev, uuid=current))
        prev = current
    return sid, out


# ---------------------------------------------------------------- 续接前：落档与检查点

def _one_line(text, limit=80):
    return " ".join(str(text or "").split())[:limit]


def checkpoint_args(events, window, me, user):
    import background_archive as ba
    names = {"user": user, "assistant": me}
    lines = []
    for event in events[-40:]:
        if event.get("isSidechain") or event.get("isMeta"):
            continue
        text = ba._clean(ba._claude_code_text(event.get("message")))
        if text and not (event["type"] == "user" and ba._HOST_EVENT_RE.match(text)):
            lines.append(f"{names[event['type']]}：{_one_line(text)}")
    topics = lines[-6:]
    recent = "；".join(topics[-3:]) or "没有可回看的近期正文"
    return {"window": window, "topics": topics,
            "current_state": f"（续接检查点，自动生成）同一个会话仍在继续，只是裁掉了较早的上下文；最近的话头：{recent}"}


def _state(state_dir, transcript):
    path = Path(state_dir).expanduser() / f"{hashlib.sha256(str(Path(transcript).parent).encode()).hexdigest()[:16]}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    data.setdefault("forges", 0)
    return path, data


def forge(cfg, transcript):
    events = load_events(transcript)
    real = last_context_tokens(events)
    if not cfg.force and (real is None or real <= cfg.threshold):
        return {"status": "skip", "contextTokens": real, "threshold": cfg.threshold}
    start = find_boundary(events, cfg.retain, real)
    if start is None:
        return {"status": "skip", "reason": "保留量以内，或者找不到干净的切点", "contextTokens": real}
    if cfg.archive:
        import background_archive as ba
        flushed = ba.flush(cfg.archive_cfg)
        if flushed["status"] != "flushed":
            return {"status": "aborted", "reason": f"落档没跑完，不裁：{flushed.get('error') or flushed['status']}"}
    state_path, state = _state(cfg.state_dir, transcript)
    window = cfg.window or state["forges"] + 1
    if cfg.url:
        import background_archive as ba
        try:
            ba.call_latent(cfg.url, cfg.token, "latent_thread_close",
                           checkpoint_args(events[:start] + events[start:], window, cfg.me, cfg.user))
        except Exception as e:
            return {"status": "aborted", "reason": f"检查点没写进 Latent，不裁：{e}"}
    kept = with_tool_primer(events, start)
    sid, rewritten = rewrite_chain(kept)
    out = Path(transcript).parent / f"{sid}.jsonl"
    tmp = out.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in rewritten), encoding="utf-8")
    os.replace(tmp, out)
    state.update(forges=state["forges"] + 1, last={"at": time.time(), "from": str(transcript), "to": str(out)})
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    return {"status": "forged", "sessionId": sid, "path": str(out), "kept": len(rewritten),
            "keptFrom": events[start].get("timestamp"), "contextTokens": real}


# ---------------------------------------------------------------- 命令行

def build_config(argv):
    env = os.environ.get
    p = argparse.ArgumentParser(description="Latent 原地续接（Claude Code 会话）")
    p.add_argument("command", choices=["check", "forge"])
    p.add_argument("transcript")
    p.add_argument("--threshold", type=int, default=int(env("LATENT_FORGE_THRESHOLD", DEFAULT_THRESHOLD)))
    p.add_argument("--retain", type=int, default=int(env("LATENT_FORGE_RETAIN", DEFAULT_RETAIN)))
    p.add_argument("--force", action="store_true", help="不看门槛，只要超过保留量就裁")
    p.add_argument("--archive", action="store_true", help="裁之前先按《后台落档》把没整理的对话收进记忆库")
    p.add_argument("--window", type=int, default=None, help="检查点的窗口号；默认是这个会话目录第几次续接")
    p.add_argument("--url", default=env("LATENT_MCP_URL"))
    p.add_argument("--token", default=env("LATENT_MCP_TOKEN"))
    p.add_argument("--me", default=env("LATENT_ARCHIVE_ME", "我"))
    p.add_argument("--user", default=env("LATENT_ARCHIVE_USER", "对方"))
    p.add_argument("--state-dir", default=env("LATENT_FORGE_STATE_DIR", "~/.cache/latent-forge"))
    cfg = p.parse_args(argv)
    if cfg.retain >= cfg.threshold:
        p.error("--retain 要小于 --threshold，否则刚续接完又越线")
    if cfg.archive:
        import background_archive as ba
        cfg.archive_cfg = ba.build_config(["flush", "--source", f"claude-code:{cfg.transcript}",
                                           "--me", cfg.me, "--user", cfg.user])
    return cfg


def main(argv):
    if argv[1:2] == ["--selftest"]:
        return _selftest()
    cfg = build_config(argv[1:])
    if cfg.command == "check":
        real = last_context_tokens(load_events(cfg.transcript))
        print(json.dumps({"contextTokens": real, "threshold": cfg.threshold,
                          "shouldForge": bool(real and real > cfg.threshold)}))
        return 0
    result = forge(cfg, cfg.transcript)
    print(json.dumps(result, ensure_ascii=False))
    return 1 if result["status"] == "aborted" else 0


# ---------------------------------------------------------------- 自检（合成夹具）

def _selftest():
    import tempfile
    import threading
    from types import SimpleNamespace
    from mcp_server import MemoryServer, make_http_server
    from session_thread import ThreadStore

    def ev(kind, content, n, usage=None, **extra):
        msg = {"role": kind, "content": content}
        if usage:
            msg["usage"] = usage
        return dict({"type": kind, "uuid": f"u{n:03d}", "parentUuid": f"u{n - 1:03d}" if n else None,
                     "sessionId": "old", "timestamp": f"2026-09-01T{n // 60:02d}:{n % 60:02d}:00Z",
                     "message": msg}, **extra)

    events, n = [], 0
    # 早期：一次成功的工具回合
    events += [ev("user", "帮我看看日历", 0),
               ev("assistant", [{"type": "thinking", "thinking": "查一下"},
                                {"type": "tool_use", "id": "t1", "name": "calendar", "input": {}}], 1),
               ev("user", [{"type": "tool_result", "tool_use_id": "t1", "content": "周六：爬山" * 100}], 2),
               ev("assistant", [{"type": "text", "text": "周六有爬山"}], 3)]
    n = 4
    for i in range(30):                 # 之后是纯聊天，每轮带 thinking
        events.append(ev("user", f"第{i}句：" + "嗯" * 300, n))
        events.append(ev("assistant", [{"type": "thinking", "thinking": "想" * 50},
                                       {"type": "text", "text": "好" * 300}], n + 1,
                         usage={"input_tokens": 10, "cache_read_input_tokens": 9000 + i,
                                "cache_creation_input_tokens": 0}))
        n += 2
    events.insert(10, {"type": "file-history-snapshot", "messageId": "x"})
    events.insert(20, ev("user", "工具半截", 999, isMeta=True))

    # 1. 用量、切点、干净边界
    real_events = [e for e in events if e.get("type") in {"user", "assistant"}]
    assert last_context_tokens(real_events) == 10 + 9029
    assert find_boundary(real_events, 20000, 9039) is None, "真实上下文没超过保留量：不裁"
    start = find_boundary(real_events, 3000, 9039)
    assert start and _is_real_user(real_events[start]) and real_events[start]["message"]["content"].startswith("第")
    # 2. 工具先例：保留区间没有工具回合 → 垫上早期那次，thinking 去掉、结果压短
    kept = with_tool_primer(real_events, start)
    assert kept[0]["message"]["content"][0]["type"] == "tool_use" and len(kept) == len(real_events) - start + 2
    assert kept[1]["message"]["content"][0]["content"].startswith("成功：") and \
        len(kept[1]["message"]["content"][0]["content"]) <= PRIMER_RESULT_CHARS + 3
    assert with_tool_primer(real_events, 0) == real_events, "保留区间自己有工具回合就不垫"
    # 3. 父链：新 id、第一条 parentUuid=None、逐条接上、uuid 不变、thinking 原样留着
    sid, chain = rewrite_chain(kept)
    assert all(e["sessionId"] == sid for e in chain) and chain[0]["parentUuid"] is None
    assert all(chain[i]["parentUuid"] == chain[i - 1]["uuid"] for i in range(1, len(chain)))
    assert [e["uuid"] for e in chain] == [e["uuid"] for e in kept]
    assert any(b.get("type") == "thinking" for e in chain[2:] if e["type"] == "assistant"
               for b in e["message"]["content"]), "保留区间的 thinking 不能丢"

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        proj = root / "projects" / "demo"
        proj.mkdir(parents=True)
        transcript = proj / "old.jsonl"
        transcript.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events) + "坏行\n",
                              encoding="utf-8")
        base = dict(threshold=9000, retain=3000, force=False, archive=False, window=None, url=None, token=None,
                    me="阿砚", user="林夏", state_dir=str(root / "state"))
        # 4. 没到门槛：什么都不写
        r = forge(SimpleNamespace(**dict(base, threshold=20000)), transcript)
        assert r["status"] == "skip" and len(list(proj.glob("*.jsonl"))) == 1, r
        # 5. 检查点写不进去：不裁
        r = forge(SimpleNamespace(**dict(base, url="http://127.0.0.1:9/mcp")), transcript)
        assert r["status"] == "aborted" and len(list(proj.glob("*.jsonl"))) == 1, r
        # 6. 真服务端：先写检查点，再写新会话文件
        store = ThreadStore(root / "threads.jsonl")
        srv = make_http_server(MemoryServer(thread_store=store), port=0, token="t")
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        r = forge(SimpleNamespace(**dict(base, url=f"http://127.0.0.1:{srv.server_address[1]}/mcp", token="t")),
                  transcript)
        srv.shutdown()
        assert r["status"] == "forged", r
        new = load_events(r["path"])
        assert Path(r["path"]).parent == proj and new[0]["sessionId"] == r["sessionId"]
        assert len(new) == r["kept"] and new[-1]["uuid"] == real_events[-1]["uuid"]
        thread = store.latest()
        assert thread.window == 1 and "续接检查点" in thread.current_state and "林夏：第29句" in "".join(thread.topics), thread
        # 7. 落档没跑完：不裁
        import background_archive as ba
        saved = ba.flush
        ba.flush = lambda cfg: {"status": "failed", "error": "模型不在"}
        try:
            r = forge(SimpleNamespace(**dict(base, archive=True, archive_cfg=None)), transcript)
        finally:
            ba.flush = saved
        assert r["status"] == "aborted" and "落档" in r["reason"], r
    print("selftest ok（forge_session：真实用量与切点 / 干净边界 / 工具先例 / 父链与 uuid / thinking 保留 / "
          "不到门槛不写 / 检查点或落档失败不裁 / 真服务端检查点）")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
