# car-command：小车语音/文本 → JSON 指令（DeepSeek 1.5B 微调 + Ollama 部署）

本文档假定环境为 **Ubuntu 服务器**（SSH 登录后在终端操作），项目路径示例为 **`~/workspace/car-command`**（请按你的实际目录替换）。

**说明**：**微调、合并、GGUF 转换与量化**均按下面各节的 **bash 命令**在服务器上顺序执行即可。

---

## 目录结构（与流程相关）

| 路径 | 说明 |
|------|------|
| `data/dataset.jsonl` | 训练用指令数据 |
| `models/DeepSeek-R1-Distill-Qwen-1.5B/` | 基座模型（本地，可与 HF 同名） |
| `final_adapter/` | 微调产出的 LoRA（§3 命令执行后生成） |
| `merged_model_fp16/` | 合并后的 HF 权重（§4 命令执行后生成） |
| `llama.cpp/` | 第三方仓库，§5 在服务器上 `git clone` 后编译与转换 |
| `deploy/Modelfile` | Ollama 模型定义 |
| `deploy/robot_q4_km.gguf` | 量化后的模型（§5 命令生成后 `cp` 到此） |
| `client/chat_loop.py` | 交互测试（推荐：SFT 模板 + `/api/generate`） |

---

## 0. Ubuntu 环境与依赖

进入项目目录：

```bash
cd ~/workspace/car-command
```

建议使用 Python 3.10–3.12 与虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**先安装 PyTorch**（按你机器的 CUDA 版本在 [pytorch.org](https://pytorch.org/get-started/locally/) 选择命令）。示例（CUDA 12.x，以官网为准）：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

**再安装项目其余依赖**（须能访问 **PyPI**，不要只用 PyTorch 索引装整份依赖）：

```bash
pip install -r requirements.txt
```

---

## 1. 基座模型准备

### 方式 A：已随项目复制 `models/DeepSeek-R1-Distill-Qwen-1.5B/`

无需额外操作，训练与合并时 `--model` / `--base` 指向该目录即可。

### 方式 B：在服务器终端下载到本地目录

```bash
cd ~/workspace/car-command
source .venv/bin/activate

python scripts/download_model.py \
  --repo deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --local-dir models/DeepSeek-R1-Distill-Qwen-1.5B
```

---

## 2. 数据准备（服务器终端）

```bash
cd ~/workspace/car-command
source .venv/bin/activate

# 基础条数
python scripts/generate_dataset.py --out data/dataset.jsonl

# 需要更多样本时（前缀扩充 + 去重，约数百条）
python scripts/generate_dataset.py --augment --out data/dataset.jsonl
```

---

## 3. Ubuntu 服务器：QLoRA 微调（终端命令）

在已激活的 Python 环境中，于**项目根目录**执行：

```bash
cd ~/workspace/car-command
source .venv/bin/activate

python scripts/train_qlora.py \
  --model models/DeepSeek-R1-Distill-Qwen-1.5B \
  --data data/dataset.jsonl \
  --output-dir ./output \
  --adapter-dir ./final_adapter
```

产出目录：`output/`、`final_adapter/`。

---

## 4. Ubuntu 服务器：合并为 FP16（终端命令）

仍在项目根目录、同一虚拟环境中执行：

```bash
cd ~/workspace/car-command
source .venv/bin/activate

python scripts/merge_lora.py \
  --base models/DeepSeek-R1-Distill-Qwen-1.5B \
  --adapter ./final_adapter \
  --out ./merged_model_fp16
```

产出目录：`merged_model_fp16/`。

---

## 5. Ubuntu 服务器：llama.cpp 编译、转 GGUF、量化（终端命令）

依赖：`git`、`cmake`、`g++`（或完整 build-essential）。**全程在终端执行**，不跑项目内 `.sh`。

```bash
cd ~/workspace/car-command
test -d llama.cpp || git clone https://github.com/ggerganov/llama.cpp.git

cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"

# 使用与训练相同的 Python 环境（含 convert 脚本依赖）
source ../.venv/bin/activate
python convert_hf_to_gguf.py ../merged_model_fp16 --outfile robot_fp16.gguf

./build/bin/llama-quantize robot_fp16.gguf robot_q4_km.gguf Q4_K_M
cp robot_q4_km.gguf ../deploy/
```

确认 `~/workspace/car-command/deploy/` 下同时有 `Modelfile` 与 `robot_q4_km.gguf`（与 `Modelfile` 里 `FROM ./robot_q4_km.gguf` 一致）。

---

## 6. Ubuntu 服务器：安装并启动 Ollama

若尚未安装：

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

创建自定义模型（在 `deploy` 目录执行，保证相对路径正确）：

```bash
cd ~/workspace/car-command/deploy
ollama create my_robot -f Modelfile
```

启动服务：

- 若系统已用 **systemd** 跑 Ollama，一般 **无需**再执行 `ollama serve`；可用 `curl http://127.0.0.1:11434/api/tags` 测试。
- 若需前台调试：`ollama serve`（端口 **11434** 被占用时，说明服务已在运行，勿重复启动）。

---

## 7. 调用与测试（服务器终端）

### 7.1 交互循环（推荐，与训练格式一致）

使用 **SFT 模板 + `/api/generate`**，每轮独立请求：

```bash
cd ~/workspace/car-command
source .venv/bin/activate
pip install requests   # 若尚未安装

python client/chat_loop.py
```

只测一句话后退出：

```bash
python client/chat_loop.py --once "跟着我"
```

---

## 流程一览（均在 Ubuntu 服务器终端按顺序执行）

1. **§0 环境**：`python3 -m venv` → `source .venv/bin/activate` → 安装 `torch` → `pip install -r requirements.txt`  
2. **§1 模型**：已有 `models/...` 或执行 §1「方式 B」中的 `python scripts/download_model.py ...`  
3. **§2 数据**：执行 §2 中的 `python scripts/generate_dataset.py ...` → 得到 `data/dataset.jsonl`  
4. **§3 微调**：执行 §3 整段命令 → 得到 `final_adapter/`  
5. **§4 合并**：执行 §4 整段命令 → 得到 `merged_model_fp16/`  
6. **§5 量化**：执行 §5 整段命令（`cmake` / `convert_hf_to_gguf.py` / `llama-quantize` / `cp`）→ 得到 `deploy/robot_q4_km.gguf`  
7. **§6 部署**：在 `deploy` 目录执行 `ollama create` → 确认 Ollama 监听 `11434`  
8. **§7 验证**：执行 `python client/chat_loop.py` 或 `curl`

---

## 常见问题（Ubuntu）

| 现象 | 处理 |
|------|------|
| `pip` 找不到 `torch` 等 | 先用 PyTorch 官网命令装 `torch`，再用默认 PyPI 装 `requirements.txt` |
| `ollama serve` 报端口占用 | 服务已在运行，直接用 API；或 `sudo systemctl stop ollama` 后再前台启动 |
| 聊天 API 输出乱、第二轮失败 | 指令解析请用 `client/chat_loop.py` 默认模式（SFT + generate），勿用 `--chat` 做主力 |

---

## 许可证与第三方

- 基座模型以 [Hugging Face - deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) 页面许可为准。  
- `llama.cpp` 为独立开源项目，请遵循其仓库许可。
