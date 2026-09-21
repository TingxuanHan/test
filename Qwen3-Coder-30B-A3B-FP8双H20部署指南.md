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

## 故障修复：安装正确的vLLM cu129 wheel

根因已经确定：

> **PyTorch使用cu129，但当前安装的vLLM wheel是CUDA 13版本。**

vLLM v0.29.0的PyPI默认包使用CUDA 13.0；CUDA 12.9需要安装release assets中单独提供的`+cu129` wheel。近期也有同类问题报告：`torch cu129`环境错误选择CUDA 13 vLLM artifact后，导入时会要求`libcudart.so.13`。([GitHub][5])

当前`torch 2.13.0+cu129`及`torch.version.cuda = 12.9`是正确的；需要重建Qwen虚拟环境并明确安装vLLM v0.29.0的官方cu129 wheel。模型和系统CUDA 12.9无需重新安装。

### 0. 重建Qwen环境并固定CUDA 12.9

模型不需要重新下载。

先固定缓存位置：

```bash
export UV_CACHE_DIR=/mnt/data/txhan/.cache/uv
export PIP_CACHE_DIR=/mnt/data/txhan/.cache/pip
export HF_HOME=/mnt/tydrive/txhan/.cache/huggingface
export TMPDIR=/mnt/data/txhan/qwen3-coder/tmp

export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
hash -r
```

确认：

```bash
nvcc --version
```

应该是：

```text
release 12.9
```

然后重建环境：

> [!CAUTION]
> 以下操作只应删除 `/mnt/data/txhan/qwen3-coder/.venv`。执行前确认当前目录正确；Qwen模型目录不在该路径中。

```bash
cd /mnt/data/txhan/qwen3-coder

deactivate 2>/dev/null || true
rm -rf .venv

uv venv .venv --python 3.10
source .venv/bin/activate
```

---

### 1. 不要再执行这条

这次不要再用：

```bash
uv pip install vllm --torch-backend=cu129
```

虽然看起来指定了 cu129，但当前 vLLM 的 PyPI/release 生态存在 CUDA variant 解析容易混到默认 CUDA 13 wheel 的情况。vLLM 官方 release 明确把 PyPI 标为 CUDA 13.0，同时另行提供 CUDA 12.9 release artifact。([GitHub][5])

---

### 2. 找到官方 v0.29.0 cu129 wheel

直接让 GitHub API 告诉我们真实下载地址，避免我硬编码文件名：

```bash
curl -s https://api.github.com/repos/vllm-project/vllm/releases/tags/v0.29.0 \
  | grep browser_download_url \
  | grep cu129 \
  | grep x86_64
```

应该返回一个类似：

```text
https://github.com/vllm-project/vllm/releases/download/v0.29.0/...
cu129...
x86_64.whl
```

如果返回多个，查看：

```bash
curl -s https://api.github.com/repos/vllm-project/vllm/releases/tags/v0.29.0 \
  | grep browser_download_url \
  | grep '\.whl' \
  | grep cu129
```

官方 v0.29.0 release 页面明确说明 CUDA 12.9 Python wheel 位于 release assets 中。([GitHub][5])

---

### 3. 自动取 x86_64 cu129 wheel URL

执行：

```bash
VLLM_CU129_WHEEL=$( \
  curl -s https://api.github.com/repos/vllm-project/vllm/releases/tags/v0.29.0 \
  | grep browser_download_url \
  | grep cu129 \
  | grep x86_64 \
  | grep '\.whl' \
  | head -1 \
  | cut -d '"' -f 4 \
)
```

检查：

```bash
echo "$VLLM_CU129_WHEEL"
```

必须看到一个：

```text
https://github.com/vllm-project/vllm/releases/download/v0.29.0/...
```

并且里面有：

```text
cu129
```

如果 `echo` 为空，先不要继续。

---

### 4. 安装真正的 cu129 vLLM

执行：

```bash
uv pip install "$VLLM_CU129_WHEEL" \
  --extra-index-url https://download.pytorch.org/whl/cu129
```

再：

```bash
uv pip install huggingface_hub
```

不要单独装 `torchaudio`。

---

### 5. 第一轮验证：Torch

```bash
python - <<'PY'
import torch

print("torch =", torch.__version__)
print("torch CUDA =", torch.version.cuda)
print("CUDA available =", torch.cuda.is_available())
print("GPU count =", torch.cuda.device_count())
PY
```

应该是：

```text
torch = 2.13.0+cu129
torch CUDA = 12.9
CUDA available = True
GPU count = 8
```

---

### 6. 再看有没有 CUDA 13 污染

执行：

```bash
uv pip list | grep -Ei 'vllm|torch|cuda|cu13'
```

这里我们主要不希望再看到明显的 CUDA 13 runtime，例如：

```text
nvidia-cuda-runtime-cu13
```

那些：

```text
nvidia-cuda-nvcc 13.x
nvidia-cuda-crt 13.x
```

在一个**全新 venv** 里如果仍然出现，也值得继续查来源；但真正决定 vLLM 是否可用的是其 `.so` 链接到哪个 libcudart。

---

### 7. 最重要：检查 vLLM wheel 链接的是 CUDA 12 还是 13

先不要直接 import。

执行：

```bash
find .venv/lib/python3.10/site-packages/vllm \
  -type f -name '*.so' -print0 |
while IFS= read -r -d '' f; do
    if readelf -d "$f" 2>/dev/null | grep -q libcudart; then
        echo "=== $f ==="
        readelf -d "$f" | grep libcudart
    fi
done
```

**正确 cu129 wheel 应该出现：**

```text
libcudart.so.12
```

不应该再看到：

```text
libcudart.so.13
```

vLLM 自己的近期 issue 也做了同样的验证：正确 cu129 artifact 依赖的是 `libcudart.so.12`，错误 fallback wheel 则依赖 `libcudart.so.13`。([GitHub][6])

---

### 8. 然后才 import vLLM

```bash
python - <<'PY'
import torch

print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)

import vllm

print("vLLM:", vllm.__version__)
print("IMPORT SUCCESS")
PY
```

目标：

```text
Torch: 2.13.0+cu129
Torch CUDA: 12.9
vLLM: 0.29.0
IMPORT SUCCESS
```

如果这里成功：

> `libcudart.so.13` 问题彻底解决。

---

### 9. 清理旧编译缓存

> [!CAUTION]
> 以下目录是共享编译缓存，可能被其他Qwen、vLLM或root用户任务使用。确认没有相关进程运行并核对目标路径后再清理；执行`sudo rm -rf /root/...`前尤其应先备份或确认其中没有其他项目需要的缓存。

```bash
rm -rf /mnt/data/txhan/.cache/vllm/*
rm -rf /mnt/data/txhan/.cache/torch_extensions/*
```

如果 `/root` 下还有以前残留：

```bash
sudo rm -rf /root/.cache/vllm
sudo rm -rf /root/.cache/torch_extensions
```

不要碰：

```text
/mnt/tydrive/txhan/models/
```

---

### 10. 单卡启动

```bash
export QWEN_MODEL_DIR=/mnt/tydrive/txhan/models/Qwen3-Coder-30B-A3B-Instruct-FP8

export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

export CUDA_VISIBLE_DEVICES=0
```

启动：

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

这次重点观察：

```text
ImportError: libcudart.so.13
```

是否彻底消失。

---

### 11. 单卡成功再双卡

```bash
export CUDA_VISIBLE_DEVICES=0,1
```

然后：

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

成功以后才改：

```bash
--max-model-len 65536
```

最后才去掉：

```bash
--enforce-eager
```

再最后才加 Agent 参数。

---

### 根因链条

你之前实际上是：

```text
CUDA Toolkit 12.9
       │
       ├── nvcc 12.9
       │
       └── PyTorch 2.13 cu129   ✅
                    │
                    │
             vLLM PyPI wheel
                    │
               CUDA 13.0 ❌
                    │
              libcudart.so.13
                    │
          系统找不到 → ImportError
```

这不是“CUDA 12.9 配错了”，而是**vLLM wheel variant 装错了**。

而且这与 vLLM 近期公开 issue 的表现高度一致：cu129 PyTorch 环境里意外选到 CUDA 13 vLLM artifact，最终就是 `libcudart.so.13`。([GitHub][7])

所以这次不要再折腾系统 CUDA，也**不要安装 CUDA 13**。核心就是：

> **明确安装 vLLM v0.29.0 的官方 cu129 release wheel，而不是默认 PyPI wheel。**

[1]: https://docs.vllm.ai/en/stable/getting_started/installation/gpu/ "GPU - vLLM"
[2]: https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8/tree/main "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 at main"
[3]: https://docs.vllm.ai/en/latest/features/tool_calling/ "Tool Calling - vLLM"
[4]: https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8/blob/main/config.json "config.json · Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 at main"
[5]: https://github.com/vllm-project/vllm/releases "Releases · vllm-project/vllm · GitHub"
[6]: https://github.com/vllm-project/vllm/issues/52544 "VLLM_USE_PRECOMPILED can fall back across CUDA variants without compatibility validation · Issue #52544 · GitHub"
[7]: https://github.com/vllm-project/vllm/issues/42338 "cu129 Nightly Installation Resolves to CUDA 13 Wheel · Issue #42338 · GitHub"
