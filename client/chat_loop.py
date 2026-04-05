"""
与 Ollama 交互调试：默认按「训练时同款」模板单次调用 /api/generate，避免 chat 模板导致乱输出或第二次失败。

训练格式（与 scripts/train_qlora.py 一致）：
  ### Instruction:
  {instruction}
  ### Input:
  {用户话}
  ### Response:

用法：
  python client/chat_loop.py
  python client/chat_loop.py --instruction "解析指令"
  python client/chat_loop.py --once "跟着我"          # 只调一次后退出
  python client/chat_loop.py --chat                   # 仍用旧版 chat/v1 回退（不推荐本任务）

环境变量：OLLAMA_MODEL, OLLAMA_BASE_URL
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import requests


def _normalize_base(url: str) -> str:
    url = url.rstrip("/")
    if not url.startswith("http"):
        url = f"http://{url}"
    return url


def build_sft_prompt(instruction: str, user_text: str) -> str:
    return (
        "### Instruction:\n"
        f"{instruction}\n"
        "### Input:\n"
        f"{user_text}\n"
        "### Response:\n"
    )


def extract_json_object(text: str) -> str | None:
    """从整段回复里抠出最后一个完整 {...}（应对思考标签、多行废话）。"""
    text = text.strip()
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass
    start = text.rfind("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _strip_think_tags(text: str) -> str:
    """若含推理结束标记，只保留其后文本（便于再抽 JSON）。"""
    if "</redacted_thinking>" in text:
        text = text.split("</think>", 1)[1].strip()
    return text


def call_generate(
    base: str,
    model: str,
    prompt: str,
    timeout: float,
) -> tuple[str | None, str]:
    url = f"{base}/api/generate"
    try:
        r = requests.post(
            url,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1},
            },
            timeout=timeout,
        )
    except requests.RequestException as e:
        return None, f"网络错误: {e}"
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}: {r.text[:500]}"
    data = r.json()
    raw = (data.get("response") or "").strip()
    if not raw:
        return None, "响应为空"
    return raw, "ok"


# ---------- 可选：旧版 chat 回退（不推荐本任务）----------


def _messages_to_prompt(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for m in messages:
        role, content = m.get("role", ""), m.get("content", "")
        if role == "user":
            parts.append(f"用户: {content}")
        elif role == "assistant":
            parts.append(f"助手: {content}")
        else:
            parts.append(f"{role}: {content}")
    parts.append("助手:")
    return "\n".join(parts)


def _chat_openai(base: str, model: str, messages: list[dict[str, str]], timeout: float) -> str | None:
    r = requests.post(
        f"{base}/v1/chat/completions",
        json={"model": model, "messages": messages, "stream": False},
        timeout=timeout,
    )
    if r.status_code != 200:
        return None
    data = r.json()
    choices = data.get("choices") or []
    if not choices:
        return None
    msg = choices[0].get("message") or {}
    return (msg.get("content") or "").strip()


def _chat_native(base: str, model: str, messages: list[dict[str, str]], timeout: float) -> str | None:
    r = requests.post(
        f"{base}/api/chat",
        json={"model": model, "messages": messages, "stream": False},
        timeout=timeout,
    )
    if r.status_code != 200:
        return None
    data = r.json()
    return ((data.get("message") or {}).get("content") or "").strip()


def _generate_plain(base: str, model: str, prompt: str, timeout: float) -> str | None:
    r = requests.post(
        f"{base}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    if r.status_code != 200:
        return None
    data = r.json()
    return (data.get("response") or "").strip()


def complete_round_chat(
    base: str,
    model: str,
    messages: list[dict[str, str]],
    timeout: float,
) -> tuple[str | None, str]:
    err_parts: list[str] = []
    content = _chat_openai(base, model, messages, timeout)
    if content:
        return content, "/v1/chat/completions"
    err_parts.append("v1/chat/completions 失败")
    content = _chat_native(base, model, messages, timeout)
    if content:
        return content, "/api/chat"
    err_parts.append("api/chat 失败")
    prompt = _messages_to_prompt(messages)
    content = _generate_plain(base, model, prompt, timeout)
    if content:
        return content, "/api/generate (拼接)"
    err_parts.append("api/generate 失败")
    return None, "; ".join(err_parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="小车指令解析调试：默认 SFT 模板 + generate")
    parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "my_robot"))
    parser.add_argument(
        "--url",
        default=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        help="Ollama 根地址",
    )
    parser.add_argument("--instruction", default="解析指令", help="与数据集 instruction 字段一致更佳")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--chat", action="store_true", help="使用 chat/v1 多轮（易与微调格式不一致）")
    parser.add_argument("--once", metavar="TEXT", help="只请求一次该句话后退出")
    args = parser.parse_args()

    base = _normalize_base(args.url)

    def handle_one(user_text: str) -> None:
        if args.chat:
            messages: list[dict[str, str]] = [{"role": "user", "content": user_text}]
            content, via = complete_round_chat(base, args.model, messages, args.timeout)
            if not content:
                print(f"失败（{via}）\n")
                return
            print(f"原始 ({via}): {content}\n")
        else:
            prompt = build_sft_prompt(args.instruction, user_text)
            content, err = call_generate(base, args.model, prompt, args.timeout)
            if not content:
                print(f"失败: {err}\n")
                return
            print(f"原始 (/api/generate + SFT 模板): {content}\n")

        cleaned = _strip_think_tags(content)
        blob = extract_json_object(cleaned) or extract_json_object(content)
        if blob:
            try:
                obj = json.loads(blob)
                print("JSON:", json.dumps(obj, ensure_ascii=False, indent=2), "\n")
            except json.JSONDecodeError:
                print("JSON 解析失败，片段:", blob[:200], "\n")
        else:
            print("未从回复中解析出 JSON 对象。\n")

    if args.once is not None:
        print(f"模型: {args.model}  |  {base}\n")
        handle_one(args.once.strip())
        return

    print(f"模型: {args.model}  |  {base}")
    if args.chat:
        print("模式: chat（多轮累积上下文）")
    else:
        print(f"模式: SFT 单次（instruction={args.instruction!r}，每轮独立，不累积 chat 历史）")
    print("输入 q 回车退出。\n")

    messages_chat: list[dict[str, str]] = []

    while True:
        try:
            line = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            sys.exit(0)

        if line.lower() == "q":
            print("再见。")
            break
        if not line:
            continue

        if args.chat:
            messages_chat.append({"role": "user", "content": line})
            try:
                content, via = complete_round_chat(base, args.model, messages_chat, args.timeout)
            except requests.RequestException as e:
                print(f"网络错误: {e}\n")
                messages_chat.pop()
                continue
            if not content:
                print(f"失败（{via}）\n")
                messages_chat.pop()
                continue
            print(f"原始 ({via}): {content}\n")
            messages_chat.append({"role": "assistant", "content": content})
            cleaned = _strip_think_tags(content)
            blob = extract_json_object(cleaned) or extract_json_object(content)
            if blob:
                try:
                    print("JSON:", json.dumps(json.loads(blob), ensure_ascii=False, indent=2), "\n")
                except json.JSONDecodeError:
                    pass
            continue

        handle_one(line)


if __name__ == "__main__":
    main()
