"""
生成小车自然语言 -> 控制 JSON 的 dataset.jsonl。

每行：{"instruction": "...", "input": "用户原话", "output": "{...}"}

运行：
  python scripts/generate_dataset.py
  python scripts/generate_dataset.py --augment          # 前缀扩充，约数百条，仍会去重
  python scripts/generate_dataset.py --out data/dataset.jsonl --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# 多种「系统侧」指令描述，增强泛化（模型需同时认不同说法）
INSTRUCTION_VARIANTS = [
    "解析指令",
    "将用户话转为小车控制 JSON",
    "识别语音里的行驶意图并输出 JSON",
    "你是车载助手，请输出动作字段",
]

# (output 字典, 用户可能说的中文短语列表)
BEHAVIORS: list[tuple[dict, list[str]]] = [
    (
        {"action": "follow", "speed": "high"},
        [
            "小车快跟上我",
            "小车跟上",
            "小车过来",
            "跟紧我",
            "赶紧跟上来",
            "快跟着我",
            "全速跟紧",
            "别掉队跟紧我",
            "靠过来",
            "贴着我走",
            "紧跟我",
            "跟上来别磨蹭",
            "追上我",
        ],
    ),
    (
        {"action": "follow", "speed": "medium"},
        [
            "跟着我",
            "跟我走",
            "保持跟在我后面",
            "慢慢跟上来就行",
            "跟在我身后开",
            "保持跟随",
            "跟车模式开一下",
            "离我近一点",
            "保持在我侧后方",
            "跟住我的速度",
        ],
    ),
    (
        {"action": "follow", "speed": "low"},
        [
            "慢慢跟着我",
            "低速跟在我后面",
            "别太快跟着我走",
        ],
    ),
    (
        {"action": "stop", "speed": "zero"},
        [
            "原地待命",
            "停下",
            "别动",
            "停止",
            "刹车",
            "停住",
            "先停这",
            "保持静止",
            "不要走了",
            "立定",
            "停一下",
            "先别开了",
            "在这等我",
            "暂停行驶",
            "驻车",
        ],
    ),
    (
        {"action": "forward", "speed": "high"},
        [
            "全速前进",
            "快点往前开",
            "猛一点往前冲",
        ],
    ),
    (
        {"action": "forward", "speed": "medium"},
        [
            "前进",
            "往前开",
            "直行",
            "继续往前走",
            "往前挪一点",
            "匀速前进",
            "一直往前",
            "朝前开",
        ],
    ),
    (
        {"action": "forward", "speed": "low"},
        [
            "慢慢往前蹭",
            "低速前进",
            "一点一点往前开",
        ],
    ),
    (
        {"action": "backward", "speed": "medium"},
        [
            "后退",
            "倒车",
            "往后退一点",
            "向后开",
        ],
    ),
    (
        {"action": "backward", "speed": "low"},
        [
            "慢慢倒车",
            "低速后退",
        ],
    ),
    (
        {"action": "turn_left", "speed": "medium"},
        [
            "左转",
            "往左拐",
            "向左转",
            "打左转",
        ],
    ),
    (
        {"action": "turn_left", "speed": "low"},
        [
            "慢慢左转",
            "缓一点左拐",
        ],
    ),
    (
        {"action": "turn_right", "speed": "medium"},
        [
            "右转",
            "往右拐",
            "向右转",
            "打右转",
        ],
    ),
    (
        {"action": "turn_right", "speed": "low"},
        [
            "慢慢右转",
            "缓一点右拐",
        ],
    ),
    (
        {"action": "return_home", "speed": "medium"},
        [
            "回充电桩",
            "回家充电",
            "返回起点",
            "自己开回去",
            "回基地",
        ],
    ),
    (
        {"action": "park", "speed": "zero"},
        [
            "靠边停",
            "泊车",
            "停到路边",
            "找地方停好",
        ],
    ),
    (
        {"action": "patrol", "speed": "low"},
        [
            "开始巡逻",
            "绕着场地转一圈",
            "巡逻模式",
            "在这附近转转",
        ],
    ),
    (
        {"action": "emergency_stop", "speed": "zero"},
        [
            "急停",
            "紧急停车",
            "立刻刹停",
            "马上停",
        ],
    ),
    (
        {"action": "slow_down", "speed": "low"},
        [
            "慢一点",
            "减速",
            "开慢点",
            "别太冲",
            "把速度降下来",
        ],
    ),
    (
        {"action": "speed_up", "speed": "high"},
        [
            "快点",
            "加速",
            "再开快一点",
            "提点速",
        ],
    ),
    (
        {"action": "honk", "speed": "zero"},
        [
            "按喇叭",
            "嘀一下",
            "提醒前面的人",
        ],
    ),
    (
        {"action": "query_status", "speed": "zero"},
        [
            "汇报一下电量",
            "现在还有多少电",
            "小车状态怎么样",
            "自检一下",
            "故障有没有",
        ],
    ),
    (
        {"action": "turn_around", "speed": "medium"},
        [
            "掉头",
            "转个身",
            "U型转弯",
            "调个头往回开",
        ],
    ),
    (
        {"action": "avoid", "speed": "low"},
        [
            "绕开前面障碍",
            "躲开那个人",
            "别撞上去",
            "从左边绕过去",
        ],
    ),
]


def _stable_json(d: dict) -> str:
    return json.dumps(d, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


AUGMENT_PREFIXES = ("请", "帮我", "麻烦你", "现在", "立刻")


def _maybe_augment(rows: list[dict], rng: random.Random) -> list[dict]:
    """在不改标签的前提下加口语前缀，扩充样本并去重。"""
    seen = {r["input"].strip() for r in rows}
    extra: list[dict] = []
    for row in rows:
        base = row["input"].strip()
        if len(base) >= 18:
            continue
        for p in AUGMENT_PREFIXES:
            if base.startswith(p):
                continue
            cand = f"{p}{base}"
            if cand in seen:
                continue
            seen.add(cand)
            extra.append(
                {
                    "instruction": row["instruction"],
                    "input": cand,
                    "output": row["output"],
                }
            )
            if rng.random() < 0.35:
                break
    rows.extend(extra)
    return rows


def build_rows(rng: random.Random) -> list[dict]:
    rows: list[dict] = []
    for spec, phrases in BEHAVIORS:
        out = _stable_json(spec)
        for i, phrase in enumerate(phrases):
            instr = INSTRUCTION_VARIANTS[(i + hash(phrase) % 997) % len(INSTRUCTION_VARIANTS)]
            rows.append({"instruction": instr, "input": phrase, "output": out})

    # 少量「带称呼/语气」的复述，同一标签增强鲁棒性
    extras: list[tuple[dict, str]] = [
        ({"action": "follow", "speed": "high"}, "喂，小车，快跟紧我"),
        ({"action": "follow", "speed": "medium"}, "小车你跟着我就行"),
        ({"action": "stop", "speed": "zero"}, "小车停下，我要拍照"),
        ({"action": "forward", "speed": "medium"}, "小车往前走"),
        ({"action": "turn_left", "speed": "medium"}, "小车左转九十度"),
        ({"action": "turn_right", "speed": "medium"}, "小车右转一点点"),
        ({"action": "return_home", "speed": "medium"}, "小车自己回充吧"),
        ({"action": "emergency_stop", "speed": "zero"}, "危险，立刻停车"),
    ]
    for spec, phrase in extras:
        rows.append(
            {
                "instruction": rng.choice(INSTRUCTION_VARIANTS),
                "input": phrase,
                "output": _stable_json(spec),
            }
        )

    rng.shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="生成小车指令解析 dataset.jsonl")
    parser.add_argument("--out", type=Path, default=Path("data/dataset.jsonl"))
    parser.add_argument("--seed", type=int, default=42, help="打乱顺序用")
    parser.add_argument(
        "--augment",
        action="store_true",
        help="为较短句子添加「请/帮我/现在」等前缀，扩充样本（仍会去重）",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    samples = build_rows(rng)
    if args.augment:
        _maybe_augment(samples, rng)
        rng.shuffle(samples)

    # 按 input 去重，保留先出现的（打乱后仍可能重复短语，保险）
    seen: set[str] = set()
    unique: list[dict] = []
    for row in samples:
        key = row["input"].strip()
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in unique:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    actions = {json.loads(x["output"])["action"] for x in unique}
    print(f"已写入 {len(unique)} 条样本到 {args.out.resolve()}")
    print(f"动作类型数: {len(actions)} -> {', '.join(sorted(actions))}")


if __name__ == "__main__":
    main()
