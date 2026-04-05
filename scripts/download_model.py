"""
从 Hugging Face 拉取基础模型到本地目录（可选，便于离线训练）。

默认模型：deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

用法：
  python scripts/download_model.py
  python scripts/download_model.py --local-dir ./models/DeepSeek-R1-Distill-Qwen-1.5B
"""

from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download


DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_MODEL, help="Hugging Face 模型 ID")
    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help="保存到该目录；不指定则仅缓存到默认 HF 缓存",
    )
    args = parser.parse_args()
    kw = {"repo_id": args.repo}
    if args.local_dir:
        kw["local_dir"] = args.local_dir
    path = snapshot_download(**kw)
    print(f"下载完成: {path}")


if __name__ == "__main__":
    main()
