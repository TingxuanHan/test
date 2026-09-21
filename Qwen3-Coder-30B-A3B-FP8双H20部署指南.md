# Qwen3-Coder-30B-A3B-Instruct-FP8 双 H20 部署指南

## 方案说明

现有 `MODEL_DIR` 已供 **DeepSeek V4 Flash** 使用，本方案保持该变量不变。Qwen 全部使用独立变量：

```bash
QWEN_MODEL_DIR
QWEN_WORKDIR
QWEN_PORT
```

这样以后 DeepSeek 和 Qwen 可以并存，路径、脚本、端口都不会冲突。

鉴于此前出现 **PyTorch cu132 / torchaudio cu130 冲突**，建议重新创建干净的 Qwen Python 环境，并明确锁定到 **CUDA 13.0**。vLLM 官方当前支持 `--torch-backend=cu130`。([vLLM][1])

---

## 一、最终目录规划

你的磁盘：

```text
/mnt/data/txhan          700 GB
/mnt/tydrive/txhan       2 TB
```

建议：

```text
/mnt/tydrive/txhan/
├── DeepSeek/
│   └── ...                        # 你现有 DeepSeek，不动
│
├── models/
│   └── Qwen3-Coder-30B-A3B-Instruct-FP8/
│
└── .cache/
    └── huggingface/

/mnt/data/txhan/
└── qwen3-coder/
    ├── .venv/
    ├── logs/
    ├── tmp/
    └── start_qwen.sh
```

原则：

```text
2TB盘 → 模型、HF缓存
700GB盘 → Python环境、日志、启动脚本
```

---

## 二、保留现有 MODEL_DIR

你现在可能已经有：

```bash
echo $MODEL_DIR
```

输出类似：

```text
/mnt/tydrive/txhan/DeepSeek/...
```

**不要改。**

以后 Qwen 单独定义：

```bash
export QWEN_MODEL_DIR=/mnt/tydrive/txhan/models/Qwen3-Coder-30B-A3B-Instruct-FP8
export QWEN_WORKDIR=/mnt/data/txhan/qwen3-coder
export QWEN_PORT=8000
```

检查：

```bash
echo $MODEL_DIR
echo $QWEN_MODEL_DIR
```

应该是两个不同路径，例如：

```text
/mnt/tydrive/txhan/DeepSeek/DeepSeek-V4-Flash
/mnt/tydrive/txhan/models/Qwen3-Coder-30B-A3B-Instruct-FP8
```

这两个变量完全可以同时存在。

---

## 三、检查 GPU

```bash
nvidia-smi
```

看 8 张 H20。

再执行：

```bash
nvidia-smi topo -m
```

我们暂时假设使用：

```text
GPU 0
GPU 1
```

如果 0、1 被占用了，就换成其他两张，例如：

```text
GPU 6
GPU 7
```

下面统一先按 `0,1` 写。

---

## 四、建立 Qwen 专属目录

```bash
mkdir -p /mnt/data/txhan/qwen3-coder
mkdir -p /mnt/data/txhan/qwen3-coder/logs
mkdir -p /mnt/data/txhan/qwen3-coder/tmp

mkdir -p /mnt/tydrive/txhan/models/Qwen3-Coder-30B-A3B-Instruct-FP8
mkdir -p /mnt/tydrive/txhan/.cache/huggingface
```

然后：

```bash
cd /mnt/data/txhan/qwen3-coder
```

定义变量：

```bash
export QWEN_WORKDIR=/mnt/data/txhan/qwen3-coder
export QWEN_MODEL_DIR=/mnt/tydrive/txhan/models/Qwen3-Coder-30B-A3B-Instruct-FP8
export HF_HOME=/mnt/tydrive/txhan/.cache/huggingface
export TMPDIR=/mnt/data/txhan/qwen3-coder/tmp
export QWEN_PORT=8000
```

检查：

```bash
echo $QWEN_WORKDIR
echo $QWEN_MODEL_DIR
echo $HF_HOME
echo $TMPDIR
```

---

## 五、重建 Qwen 专属虚拟环境

因为你已经碰到：

```text
PyTorch CUDA 13.2
torchaudio CUDA 13.0
```

我建议不要修补这个旧环境，直接删掉。

如果当前环境已经激活：

```bash
deactivate 2>/dev/null || true
```

进入目录：

```bash
cd /mnt/data/txhan/qwen3-coder
```

删除**仅 Qwen 的虚拟环境**：

> [!CAUTION]
> 执行前确认当前目录为 `/mnt/data/txhan/qwen3-coder`。以下命令只应删除该目录下的 `.venv`。

```bash
rm -rf .venv
```

这不会碰：

```text
DeepSeek
Qwen模型
HF缓存
```

只删 Python 环境。

---

## 六、确认 uv

```bash
uv --version
```

如果正常，例如：

```text
uv 0.12.x
```

就继续。

建议更新一次：

```bash
uv self update
```

---

## 七、建立 Python 3.10 环境

```bash
cd /mnt/data/txhan/qwen3-coder
```

执行：

```bash
uv venv .venv --python 3.10
```

然后：

```bash
source .venv/bin/activate
```

检查：

```bash
which python
python --version
```

应该类似：

```text
/mnt/data/txhan/qwen3-coder/.venv/bin/python
Python 3.10.x
```

如果系统没有 Python 3.10：

```bash
uv python install 3.10
```

然后重新：

```bash
uv venv .venv --python 3.10
source .venv/bin/activate
```

---

## 八、安装 vLLM：明确指定 CUDA 13.0

这次**不要再用**：

```bash
--torch-backend=auto
```

因为刚才它把你的环境带到了 CUDA 13.2 组合。

改成：

```bash
uv pip install -U vllm --torch-backend=cu130
```

vLLM 官方当前明确支持这种方式指定 CUDA 13.0 backend。([vLLM][1])

然后：

```bash
uv pip install -U huggingface_hub
```

目前不需要主动装：

```text
torchaudio
torchvision
```

Qwen3-Coder 是纯文本模型。

---

## 九、检查 PyTorch / CUDA

执行：

```bash
python - <<'PY'
import torch

print("PyTorch version :", torch.__version__)
print("PyTorch CUDA    :", torch.version.cuda)
print("CUDA available  :", torch.cuda.is_available())
print("GPU count       :", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

重点应该看到：

```text
PyTorch CUDA : 13.0
CUDA available : True
GPU count : 8
```

然后：

```bash
nvidia-smi
```

这里顶部哪怕显示：

```text
CUDA Version: 13.2
```

也**没问题**。

这是：

```text
NVIDIA Driver
  └─ 最高支持 CUDA 13.2

PyTorch
  └─ 使用 CUDA 13.0 runtime
```

两者并不冲突。

---

## 十、确认无残留 torchaudio 冲突

执行：

```bash
uv pip list | grep -E 'torch|vllm'
```

如果出现：

```text
torchaudio
```

再检查：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

对于这个 Qwen 环境，实际上不需要 torchaudio。

如果它又引发版本冲突，直接：

```bash
uv pip uninstall torchaudio
```

然后：

```bash
python -c "import vllm; print(vllm.__version__)"
```

只要这里正常：

```text
vLLM ...
```

就可以继续。

---

## 十一、检查 Hugging Face 配置

你之前可能已经在 `.bashrc` 写了：

```bash
HF_HOME=/mnt/tydrive/txhan/.cache/huggingface
HF_HUB_DISABLE_XET=1
```

确认：

```bash
echo $HF_HOME
echo $HF_HUB_DISABLE_XET
```

我们不需要为了 Qwen 改 DeepSeek 正在使用的配置。

尤其是：

```text
MODEL_DIR
```

完全不碰。

---

## 十二、下载 Qwen FP8

我们下载：

```text
Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8
```

这是 Qwen 官方 FP8 checkpoint，也明确提供 vLLM 使用入口。([Hugging Face][2])

先定义：

```bash
export QWEN_MODEL_DIR=/mnt/tydrive/txhan/models/Qwen3-Coder-30B-A3B-Instruct-FP8
```

下载：

```bash
hf download \
  Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
  --local-dir "$QWEN_MODEL_DIR"
```

注意这里已经完全没有：

```bash
$MODEL_DIR
```

所以不会碰 DeepSeek。

---

## 十三、如果下载中断

直接重新执行同一条：

```bash
hf download \
  Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
  --local-dir "$QWEN_MODEL_DIR"
```

不要：

```bash
rm -rf "$QWEN_MODEL_DIR"
```

Hugging Face 会利用已有下载内容。

---

## 十四、检查模型

下载完成：

```bash
du -sh "$QWEN_MODEL_DIR"
```

然后：

```bash
ls -lh "$QWEN_MODEL_DIR"
```

重点确认存在：

```text
config.json
tokenizer.json
tokenizer_config.json
model.safetensors.index.json
model-00001-of-00004.safetensors
...
```

官方仓库确实就是这个 Qwen3-Coder FP8 模型。([Hugging Face][2])

---

## 十五、首次启动：暂不启用 Agent 工具调用

这是非常重要的一步。

第一次只验证：

```text
CUDA
↓
FP8
↓
TP=2
↓
vLLM
↓
Qwen
```

执行：

```bash
cd /mnt/data/txhan/qwen3-coder
source .venv/bin/activate

export QWEN_MODEL_DIR=/mnt/tydrive/txhan/models/Qwen3-Coder-30B-A3B-Instruct-FP8
export CUDA_VISIBLE_DEVICES=0,1
export HF_HOME=/mnt/tydrive/txhan/.cache/huggingface
export TMPDIR=/mnt/data/txhan/qwen3-coder/tmp

vllm serve "$QWEN_MODEL_DIR" \
  --served-model-name qwen3-coder \
  --tensor-parallel-size 2 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 \
  --port 8000
```

Qwen 官方 checkpoint 支持直接用 vLLM 启动。([Hugging Face][2])

---

## 十六、在另一终端观察 GPU

```bash
watch -n 1 nvidia-smi
```

应该主要看到：

```text
GPU0    vLLM
GPU1    vLLM
```

GPU2～7 不应被这个实例占用。

注意：

```bash
CUDA_VISIBLE_DEVICES=0,1
```

之后 vLLM 内部看到的是逻辑：

```text
cuda:0
cuda:1
```

这是正常的。

---

## 十七、确认 API 启动

看到类似：

```text
Application startup complete
```

另开终端：

```bash
curl http://127.0.0.1:8000/v1/models
```

应该看到：

```text
qwen3-coder
```

---

## 十八、测试生成

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-coder",
    "messages": [
      {
        "role": "user",
        "content": "Write a Python script that prints all NVIDIA GPUs and their memory usage using nvidia-smi."
      }
    ],
    "temperature": 0.2,
    "max_tokens": 1024
  }'
```

如果返回代码：

```text
H20
↓
vLLM
↓
Qwen
↓
OpenAI API
```

已经通了。

---

## 十九、启用 Agent 工具调用

普通生成确认无误以后：

```text
Ctrl+C
```

重新启动：

```bash
vllm serve "$QWEN_MODEL_DIR" \
  --served-model-name qwen3-coder \
  --tensor-parallel-size 2 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.85 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --host 0.0.0.0 \
  --port 8000
```

当前 vLLM 官方文档明确列出：

```text
Qwen3-Coder-30B-A3B-Instruct
→ qwen3_xml
```

作为 tool-call parser。([vLLM][3])

所以这里不要写旧的：

```text
qwen3_coder
```

而是：

```text
qwen3_xml
```

---

## 二十、建立最终启动脚本

这样以后根本不用记变量。

创建：

```bash
nano /mnt/data/txhan/qwen3-coder/start_qwen.sh
```

内容：

```bash
#!/usr/bin/env bash
set -e

QWEN_WORKDIR=/mnt/data/txhan/qwen3-coder
QWEN_MODEL_DIR=/mnt/tydrive/txhan/models/Qwen3-Coder-30B-A3B-Instruct-FP8
QWEN_PORT=8000

source "$QWEN_WORKDIR/.venv/bin/activate"

export CUDA_VISIBLE_DEVICES=0,1
export HF_HOME=/mnt/tydrive/txhan/.cache/huggingface
export TMPDIR="$QWEN_WORKDIR/tmp"

exec vllm serve "$QWEN_MODEL_DIR" \
  --served-model-name qwen3-coder \
  --tensor-parallel-size 2 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.85 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --host 0.0.0.0 \
  --port "$QWEN_PORT"
```

注意这里甚至没有：

```text
MODEL_DIR
```

所以 DeepSeek 的变量完全不会污染 Qwen。

保存：

```text
Ctrl+O
Enter
Ctrl+X
```

加权限：

```bash
chmod +x /mnt/data/txhan/qwen3-coder/start_qwen.sh
```

以后：

```bash
/mnt/data/txhan/qwen3-coder/start_qwen.sh
```

即可。

---

## 二十一、使用 tmux 常驻运行

```bash
tmux new -s qwen
```

然后：

```bash
/mnt/data/txhan/qwen3-coder/start_qwen.sh
```

退出但不停止：

```text
Ctrl+B
D
```

回来：

```bash
tmux attach -t qwen
```

这样 SSH 断线，Qwen 不会停。

---

## 二十二、为 DeepSeek 和 Qwen 分配独立端口

以后你的 DeepSeek 也启动以后，不建议都抢：

```text
8000
```

建议固定：

```text
Qwen       → 8000
DeepSeek   → 8001
```

或者：

```text
Qwen       → 8001
DeepSeek   → 8000
```

比如我建议：

```bash
QWEN_PORT=8001
```

那么 Qwen 脚本最后改：

```bash
--port 8001
```

这样以后：

```text
http://server:8000/v1 → DeepSeek
http://server:8001/v1 → Qwen
```

管理起来最清晰。

**如果 DeepSeek 现在还只是在下载，没有启动服务，那么 Qwen 暂时用 8000 完全没问题。**

---

## 二十三、Windows 访问服务器

如果服务器 IP 不直接开放 8000，建议走 SSH tunnel。

Windows PowerShell：

```powershell
ssh -N -L 8000:127.0.0.1:8000 用户名@服务器IP
```

保持这个窗口开着。

再开 PowerShell：

```powershell
curl.exe http://127.0.0.1:8000/v1/models
```

应该出现：

```text
qwen3-coder
```

---

## 二十四、接入 Pi

你 Windows 上的 Pi 配置：

```text
C:\Users\TxHan\.pi\agent\models.json
```

可以添加：

```json
{
  "providers": {
    "qwen-local": {
      "baseUrl": "http://127.0.0.1:8000/v1",
      "api": "openai-completions",
      "apiKey": "local",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false,
        "maxTokensField": "max_tokens"
      },
      "models": [
        {
          "id": "qwen3-coder",
          "name": "Qwen3-Coder 30B A3B FP8 - 2xH20",
          "reasoning": false,
          "input": ["text"],
          "contextWindow": 65536,
          "maxTokens": 8192,
          "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0
          }
        }
      ]
    }
  }
}
```

启动：

```powershell
pi
```

然后：

```text
/model
```

选择：

```text
Qwen3-Coder 30B A3B FP8 - 2xH20
```

---

## 二十五、进行真实 Coding Agent 测试

不要只问：

```text
你好
```

进入一个 Git 项目：

```powershell
cd D:\你的项目
pi
```

然后让它：

```text
Inspect this repository first.
Run the existing tests.
Identify the current failures and explain the likely cause.
Do not modify files yet.
```

你要观察它是不是能：

```text
读取 repo
   ↓
调用 terminal
   ↓
运行 git / pytest / npm / python
   ↓
读取 stdout/stderr
   ↓
继续分析
```

如果可以，这才叫整个 Agent 链路跑通。

---

## 二十六、稳定后将上下文提升至 128K

先保持：

```bash
--max-model-len 65536
```

跑稳定以后改：

```bash
--max-model-len 131072
```

Pi 同时改：

```json
"contextWindow": 131072
```

Qwen3-Coder 这个模型本身支持更长上下文，但你现在的目标首先是**稳定把 Pi + Qwen + H20 跑通**，没有必要第一步就把 KV Cache 拉到最大。([Hugging Face][4])

---

## 当前建议的执行起点

因为你已经：

* 安装了 `uv`
* 建过一次环境
* 但碰到了 Torch/TorchAudio CUDA 版本冲突
* `MODEL_DIR` 已经给 DeepSeek 用了

所以**不用从第一步重新折腾系统**。你现在从这里开始执行即可：

```bash
cd /mnt/data/txhan/qwen3-coder

deactivate 2>/dev/null || true
rm -rf .venv

uv self update

uv venv .venv --python 3.10
source .venv/bin/activate

uv pip install -U vllm --torch-backend=cu130
uv pip install -U huggingface_hub
```

然后：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
PY
```

**先确认这里显示 `CUDA: 13.0`，再往后下载 Qwen。**

而以后整个方案最关键的变量规范就是：

```text
MODEL_DIR          → 留给你现有 DeepSeek，不碰

QWEN_MODEL_DIR     → Qwen专用
QWEN_WORKDIR       → Qwen专用
QWEN_PORT          → Qwen专用
```

这样即使未来你再部署 Gemma，也建议继续采用：

```text
GEMMA_MODEL_DIR
GEMMA_WORKDIR
GEMMA_PORT
```

不会再发生模型之间环境变量互相覆盖的问题。

---

## 故障修复：统一到 CUDA 12.9

本方案不再继续修补当前混杂的 CUDA 环境，而是将 Qwen 专用工具链统一为：

> **系统 `nvcc` 12.9 + PyTorch cu129 + vLLM cu129。**

机器已经安装 `/usr/local/cuda-12.9`，而vLLM当前预编译CUDA二进制默认使用CUDA 12.9，因此这条路线的工具链匹配最直接。([vLLM][1])

同时将Qwen相关缓存移出`/root`，避免继续占用根分区。DeepSeek现有的`MODEL_DIR`及模型文件保持不变。

### 0. 最终目录规划

保持：

```text
/mnt/data/txhan                  # 700G
└── qwen3-coder/
    ├── .venv
    ├── logs
    └── tmp

/mnt/data/txhan/.cache/
├── uv
├── pip
├── vllm
└── torch_extensions

/mnt/tydrive/txhan              # 2TB
├── models/
│   └── Qwen3-Coder-30B-A3B-Instruct-FP8
└── .cache/
    └── huggingface
```

DeepSeek 原来的：

```bash
MODEL_DIR
```

完全不碰。

Qwen 只使用：

```bash
QWEN_MODEL_DIR
```

---

### 1. 先验证 CUDA 12.9 是正常的

不要修改任何东西，先运行：

```bash
/usr/local/cuda-12.9/bin/nvcc --version
```

应该看到：

```text
Cuda compilation tools, release 12.9
```

再：

```bash
/usr/local/cuda-12.9/bin/nvcc --help | grep -A3 compress-mode
```

应该能找到：

```text
--compress-mode
```

CUDA 12.9 官方 `nvcc` 明确支持：

```text
--compress-mode={default|size|speed|balance|none}
```

([NVIDIA Docs][5])

为了彻底验证，再实际编译一次：

```bash
cat >/tmp/test_cuda.cu <<'EOF'
__global__ void kernel() {}

int main() {
    kernel<<<1,1>>>();
    return 0;
}
EOF
```

然后：

```bash
/usr/local/cuda-12.9/bin/nvcc \
    --compress-mode=size \
    /tmp/test_cuda.cu \
    -o /tmp/test_cuda
```

如果没有报错：

```bash
ls -lh /tmp/test_cuda
```

说明 CUDA 12.9 编译器正常。

---

### 2. 把当前 shell 固定到 CUDA 12.9

执行：

```bash
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

hash -r
```

验证：

```bash
echo "$CUDA_HOME"
which nvcc
nvcc --version
```

必须对应：

```text
/usr/local/cuda-12.9

/usr/local/cuda-12.9/bin/nvcc

release 12.9
```

如果这里仍然出现 12.4，先不要继续。

---

### 3. 把所有缓存移出 `/root`

这是这次我建议你一起修掉的东西。

创建：

```bash
mkdir -p /mnt/data/txhan/.cache/uv
mkdir -p /mnt/data/txhan/.cache/pip
mkdir -p /mnt/data/txhan/.cache/vllm
mkdir -p /mnt/data/txhan/.cache/torch_extensions

mkdir -p /mnt/tydrive/txhan/.cache/huggingface

mkdir -p /mnt/data/txhan/qwen3-coder/tmp
mkdir -p /mnt/data/txhan/qwen3-coder/logs
```

设置：

```bash
export UV_CACHE_DIR=/mnt/data/txhan/.cache/uv
export PIP_CACHE_DIR=/mnt/data/txhan/.cache/pip
export VLLM_CACHE_ROOT=/mnt/data/txhan/.cache/vllm
export TORCH_EXTENSIONS_DIR=/mnt/data/txhan/.cache/torch_extensions

export HF_HOME=/mnt/tydrive/txhan/.cache/huggingface

export TMPDIR=/mnt/data/txhan/qwen3-coder/tmp
```

vLLM 官方支持通过：

```text
VLLM_CACHE_ROOT
```

改变默认的 `~/.cache/vllm`，包括大量编译缓存；默认情况下这类东西本来会继续进入 `/root/.cache/vllm`。([vLLM][6])

检查：

```bash
echo $UV_CACHE_DIR
echo $PIP_CACHE_DIR
echo $VLLM_CACHE_ROOT
echo $TORCH_EXTENSIONS_DIR
echo $HF_HOME
```

---

### 4. 删除当前混杂的 Qwen Python 环境

你之前这个 `.venv` 经历了：

```text
cu132
↓
torchaudio cu130
↓
PyTorch cu130
↓
系统 nvcc 12.4
```

不值得继续修补。

模型本身不会被删除。

> [!CAUTION]
> 以下操作只应删除 `/mnt/data/txhan/qwen3-coder/.venv`。执行前确认当前目录正确，避免误删其他环境。

执行：

```bash
cd /mnt/data/txhan/qwen3-coder

deactivate 2>/dev/null || true

rm -rf .venv
```

确认模型还在：

```bash
du -sh /mnt/tydrive/txhan/models/Qwen3-Coder-30B-A3B-Instruct-FP8
```

---

### 5. 创建全新的 Python 环境

```bash
cd /mnt/data/txhan/qwen3-coder

uv venv .venv --python 3.10
```

激活：

```bash
source .venv/bin/activate
```

检查：

```bash
which python
python --version
```

应该类似：

```text
/mnt/data/txhan/qwen3-coder/.venv/bin/python

Python 3.10.x
```

---

### 6. 安装 CUDA 12.9 对应的 vLLM

关键是这次不要：

```bash
--torch-backend=auto
```

也不要：

```bash
--torch-backend=cu130
```

而是：

```bash
uv pip install -U vllm --torch-backend=cu129
```

然后：

```bash
uv pip install -U huggingface_hub
```

当前 vLLM 官方默认预编译 CUDA 版本就是 **12.9**，因此这次工具链是对齐的。([vLLM][1])

---

### 7. 做完整一致性检查

这个步骤非常重要。

执行：

```bash
python - <<'PY'
import torch
import vllm

print("Torch version :", torch.__version__)
print("Torch CUDA    :", torch.version.cuda)
print("vLLM version  :", vllm.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count     :", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

你要看到：

```text
Torch CUDA : 12.9
CUDA available: True
GPU count: 8
```

然后：

```bash
echo $CUDA_HOME
which nvcc
nvcc --version
```

最后必须形成：

```text
PyTorch CUDA      12.9
vLLM              cu129环境
CUDA_HOME         /usr/local/cuda-12.9
nvcc              12.9
```

这才是这次修复的核心。

---

### 8. 顺便检查 Torch 三件套

执行：

```bash
uv pip list | grep -E 'torch|vllm'
```

如果存在：

```text
torch
torchvision
torchaudio
```

测试：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__, torch.version.cuda)

try:
    import torchvision
    print("torchvision:", torchvision.__version__)
except Exception as e:
    print("torchvision error:", e)

try:
    import torchaudio
    print("torchaudio:", torchaudio.__version__)
except Exception as e:
    print("torchaudio error:", e)
PY
```

如果又出现类似：

```text
PyTorch CUDA 12.9
torchaudio CUDA xx.x
```

那么 Qwen 是纯文本模型，可以先：

```bash
uv pip uninstall torchaudio
```

再验证：

```bash
python -c "import vllm; print(vllm.__version__)"
```

不要为了 Qwen 单独折腾音频依赖。

---

### 9. 清理旧的失败编译缓存

现在我们已经把新缓存转移了，但旧缓存可能还存在。

查看：

```bash
du -sh ~/.cache/vllm 2>/dev/null
du -sh ~/.cache/torch_extensions 2>/dev/null
du -sh ~/.triton 2>/dev/null
```

> [!CAUTION]
> `~/.cache/vllm`、`~/.cache/torch_extensions` 和 `~/.triton` 是当前用户的共享缓存，可能被其他项目使用。确认没有相关任务运行并核对内容后再清理；必要时先移动到备份目录。

确认它们仅为旧的失败编译缓存后，可以清理：

```bash
rm -rf ~/.cache/vllm
rm -rf ~/.cache/torch_extensions
rm -rf ~/.triton
```

不要删：

```text
/mnt/tydrive/txhan/models/Qwen...
```

---

### 10. 定义 Qwen 独立变量

不要使用 `MODEL_DIR`：

```bash
export QWEN_MODEL_DIR=/mnt/tydrive/txhan/models/Qwen3-Coder-30B-A3B-Instruct-FP8
```

检查：

```bash
echo "$MODEL_DIR"
echo "$QWEN_MODEL_DIR"
```

两者应该互不影响。

---

### 11. 第一次只做单卡最小测试

这里不要直接上：

```text
2卡
64K
tool calling
prefix caching
```

先验证最基本链路。

```bash
export CUDA_VISIBLE_DEVICES=0
```

再启动：

```bash
vllm serve "$QWEN_MODEL_DIR" \
    --served-model-name qwen3-coder \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.80 \
    --enforce-eager \
    --host 0.0.0.0 \
    --port 8001
```

注意我这里建议：

```text
Qwen      8001
DeepSeek  8000
```

这样以后两个模型不会撞端口。

---

### 12. 观察原来的 nvcc 错误是否消失

原来是：

```text
nvcc fatal:
Unknown option '--compress-mode=size'

ninja: build stopped

Worker proc VllmWorker-0 died unexpectedly
Engine core initialization failed
```

现在最关键的是确认：

```text
Unknown option '--compress-mode=size'
```

**彻底消失。**

如果 CUDA 编译过程正常继续，那么这次主要问题就修掉了。

---

### 13. 单卡成功以后测试 API

另一个终端：

```bash
curl http://127.0.0.1:8001/v1/models
```

应该能看到：

```text
qwen3-coder
```

然后：

```bash
curl http://127.0.0.1:8001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "qwen3-coder",
      "messages": [
        {
          "role": "user",
          "content": "Write a Python hello world program."
        }
      ],
      "max_tokens": 256
    }'
```

能正常返回代码，说明：

```text
CUDA 12.9
 ↓
PyTorch cu129
 ↓
vLLM
 ↓
H20
 ↓
Qwen FP8
```

全部工作。

---

### 14. 再测试两卡

停止单卡：

```text
Ctrl+C
```

然后：

```bash
export CUDA_VISIBLE_DEVICES=0,1
```

第一次双卡仍然保持 8K：

```bash
vllm serve "$QWEN_MODEL_DIR" \
    --served-model-name qwen3-coder \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.80 \
    --enforce-eager \
    --host 0.0.0.0 \
    --port 8001
```

如果启动成功：

> TP=2 也正常。

---

### 15. 然后再恢复 64K

停止：

```text
Ctrl+C
```

改成：

```bash
vllm serve "$QWEN_MODEL_DIR" \
    --served-model-name qwen3-coder \
    --tensor-parallel-size 2 \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --host 0.0.0.0 \
    --port 8001
```

确认稳定。

---

### 16. 然后去掉 `--enforce-eager`

`--enforce-eager` 是我们诊断时用的。

稳定后再试：

```bash
vllm serve "$QWEN_MODEL_DIR" \
    --served-model-name qwen3-coder \
    --tensor-parallel-size 2 \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.85 \
    --host 0.0.0.0 \
    --port 8001
```

这时候 vLLM 可以正常使用 CUDA Graph / 编译优化，性能会更好。

---

### 17. 最后才打开 Tool Calling

普通模型完全稳定以后：

```bash
vllm serve "$QWEN_MODEL_DIR" \
    --served-model-name qwen3-coder \
    --tensor-parallel-size 2 \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.85 \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --host 0.0.0.0 \
    --port 8001
```

这才是最终给 Pi/Coding Agent 用的版本。

---

### 18. 如果 TP=2 才出错

如果出现：

```text
单卡正常
双卡异常
```

但已经**没有 nvcc compress-mode 错误**，那么这就是第二个独立问题——多卡通信。

这时候再测试：

```bash
vllm serve "$QWEN_MODEL_DIR" \
    --served-model-name qwen3-coder \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.80 \
    --enforce-eager \
    --disable-custom-all-reduce \
    --host 0.0.0.0 \
    --port 8001
```

也就是说排错顺序必须是：

```text
CUDA compiler
     ↓
单 GPU
     ↓
双 GPU
     ↓
长 context
     ↓
编译优化
     ↓
Tool calling
```

不要同时改五个变量。

---

### 19. 最终启动脚本

等上面全部验证成功以后，创建：

```bash
nano /mnt/data/txhan/qwen3-coder/start_qwen.sh
```

最终内容：

```bash
#!/usr/bin/env bash
set -e

QWEN_WORKDIR=/mnt/data/txhan/qwen3-coder
QWEN_MODEL_DIR=/mnt/tydrive/txhan/models/Qwen3-Coder-30B-A3B-Instruct-FP8
QWEN_PORT=8001

# Python environment
source "$QWEN_WORKDIR/.venv/bin/activate"

# CUDA 12.9
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# GPUs
export CUDA_VISIBLE_DEVICES=0,1

# Keep caches away from /
export UV_CACHE_DIR=/mnt/data/txhan/.cache/uv
export PIP_CACHE_DIR=/mnt/data/txhan/.cache/pip
export VLLM_CACHE_ROOT=/mnt/data/txhan/.cache/vllm
export TORCH_EXTENSIONS_DIR=/mnt/data/txhan/.cache/torch_extensions

export HF_HOME=/mnt/tydrive/txhan/.cache/huggingface
export TMPDIR="$QWEN_WORKDIR/tmp"

# Model server
exec vllm serve "$QWEN_MODEL_DIR" \
    --served-model-name qwen3-coder \
    --tensor-parallel-size 2 \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.85 \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --host 0.0.0.0 \
    --port "$QWEN_PORT"
```

保存后：

```bash
chmod +x /mnt/data/txhan/qwen3-coder/start_qwen.sh
```

以后就：

```bash
/mnt/data/txhan/qwen3-coder/start_qwen.sh
```

---

### 修复原理总结

你原来的状态其实是：

```text
系统 Toolkit:
CUDA 12.4

Python:
PyTorch cu130

vLLM:
cu130

JIT CUDA compilation:
调用系统 nvcc
        ↓
编译参数/Toolkit环境出现不一致
        ↓
--compress-mode=size 报错
        ↓
Worker死亡
        ↓
EngineCore初始化失败
        ↓
shared_memory清理警告
```

`shared_memory` 那个错误完全是**后果**，不是根因。

现在统一为：

```text
H20 Driver
    │
    ├── 系统 CUDA Toolkit 12.9
    │        └── nvcc 12.9
    │
    └── Qwen .venv
             ├── PyTorch cu129
             └── vLLM cu129
```

而且当前 vLLM 官方默认就是 CUDA 12.9，因此这比为了 Qwen 再安装一个 CUDA 13.0 Toolkit 更干净。([vLLM][1])

**你现在实际上只需要从第 1 步做到第 7 步，然后先跑第 11 步的单卡测试。** 一旦单卡能启动，再继续双卡，不需要再回头折腾 CUDA。

[1]: https://docs.vllm.ai/en/stable/getting_started/installation/gpu/ "GPU - vLLM"
[2]: https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8/tree/main "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 at main"
[3]: https://docs.vllm.ai/en/latest/features/tool_calling/ "Tool Calling - vLLM"
[4]: https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8/blob/main/config.json "config.json · Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 at main"
[5]: https://docs.nvidia.com/cuda/archive/12.9.0/cuda-compiler-driver-nvcc/index.html "NVIDIA CUDA Compiler Driver 12.9 documentation"
[6]: https://docs.vllm.ai/en/v0.29.0/usage/security/ "Security - vLLM v0.29.0"
