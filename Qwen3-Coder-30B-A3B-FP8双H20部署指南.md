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

## 故障修复：通过固定URL安装vLLM cu129 wheel

GitHub API自动获取wheel地址的命令没有返回资产列表，因此不再动态解析URL，改为直接使用已确定的v0.29.0 cu129 x86_64 wheel：

```text
vllm-0.29.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl
```

该文件约548.5MB。([SourceForge][5]) 下载地址指向vLLM官方GitHub release，不需要更改CUDA Toolkit，也不需要重建现有`.venv`。

### 1. 确认服务器架构

```bash
uname -m
```

H20服务器通常应返回：

```text
x86_64
```

只有确认是`x86_64`后，才继续使用下面的wheel。

### 2. 将wheel下载到700GB数据盘

```bash
mkdir -p /mnt/data/txhan/qwen3-coder/wheels
cd /mnt/data/txhan/qwen3-coder/wheels
```

优先使用编码了`+`号的URL：

```bash
wget -c \
'https://github.com/vllm-project/vllm/releases/download/v0.29.0/vllm-0.29.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl'
```

如果`wget`不识别`%2B`，可改用：

```bash
wget -c \
'https://github.com/vllm-project/vllm/releases/download/v0.29.0/vllm-0.29.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl'
```

下载完成后检查：

```bash
ls -lh
```

应看到大约：

```text
549M vllm-0.29.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl
```

### 3. 先解决`packaging<=24.1`冲突

激活现有环境：

```bash
cd /mnt/data/txhan/qwen3-coder
source .venv/bin/activate
```

明确从官方PyPI安装新版`packaging`：

```bash
uv pip install \
  --default-index https://pypi.org/simple \
  'packaging>=24.2'
```

检查：

```bash
python -c "import packaging; print(packaging.__version__)"
```

版本不低于24.2即可。

### 4. 直接安装本地cu129 wheel

不要再让`uv`自动选择vLLM wheel，直接安装刚下载的本地文件：

```bash
uv pip install \
  /mnt/data/txhan/qwen3-coder/wheels/vllm-0.29.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl \
  --default-index https://pypi.org/simple \
  --index https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match
```

三个来源的作用分别是：

```text
本地wheel
→ 强制vLLM使用cu129

PyPI
→ 提供packaging、flashinfer等普通依赖

PyTorch cu129 index
→ 提供torch及CUDA相关Python wheel
```

vLLM v0.29.0的默认PyPI wheel使用CUDA 13，而cu129 wheel作为release asset单独提供，因此这里固定本地asset，避免重新安装cu13版本。([GitHub][6])

### 5. 依赖仍冲突时分两步安装

如果仍卡在`packaging`解析，先安装普通依赖：

```bash
uv pip install \
  --default-index https://pypi.org/simple \
  'packaging>=24.2' \
  'flashinfer-python==0.6.18'
```

检查：

```bash
uv pip list | grep -E 'packaging|flashinfer'
```

然后再次安装本地wheel：

```bash
uv pip install \
  /mnt/data/txhan/qwen3-coder/wheels/vllm-0.29.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl \
  --default-index https://pypi.org/simple \
  --index https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match
```

### 6. 安装后检查包版本

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

应继续保持：

```text
2.13.0+cu129 12.9
```

再检查相关包：

```bash
uv pip list | grep -Ei 'vllm|torch|flashinfer|packaging'
```

重点应看到：

```text
vllm               0.29.0+cu129
torch              2.13.0+cu129
flashinfer-python  0.6.18
packaging          >=24.2
```

### 7. 启动前先导入vLLM

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

目标输出：

```text
Torch: 2.13.0+cu129
Torch CUDA: 12.9
vLLM: 0.29.0+cu129
IMPORT SUCCESS
```

如果导入成功，之前的`libcudart.so.13`问题应已消失。

### 8. 确认wheel没有链接CUDA 13

```bash
find /mnt/data/txhan/qwen3-coder/.venv/lib/python3.10/site-packages/vllm \
  -type f -name '*.so' -print0 |
while IFS= read -r -d '' f; do
    x=$(readelf -d "$f" 2>/dev/null | grep libcudart || true)
    if [ -n "$x" ]; then
        echo "=== $f ==="
        echo "$x"
    fi
done
```

正常cu129 wheel应链接：

```text
libcudart.so.12
```

不应再看到：

```text
libcudart.so.13
```

### 9. 停止使用自动URL发现

无需再执行：

```bash
VLLM_CU129_WHEEL=$(curl ...)
```

直接固定使用：

```text
vllm-0.29.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl
```

wheel从vLLM GitHub release URL下载；镜像页也列出了相同文件。([SourceForge][5])

如果第4步仍出现依赖冲突，请保留从`× No solution found`开始到结尾的完整错误输出。此时无需再修改CUDA Toolkit，剩余问题仅位于Python依赖解析层。

[1]: https://docs.vllm.ai/en/stable/getting_started/installation/gpu/ "GPU - vLLM"
[2]: https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8/tree/main "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 at main"
[3]: https://docs.vllm.ai/en/latest/features/tool_calling/ "Tool Calling - vLLM"
[4]: https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8/blob/main/config.json "config.json · Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 at main"
[5]: https://sourceforge.net/projects/vllm.mirror/files/v0.29.0/ "vLLM v0.29.0 files · SourceForge"
[6]: https://github.com/vllm-project/vllm/releases "Releases · vllm-project/vllm · GitHub"
