# Qwen3-Coder-30B-A3B-Instruct-FP8双H20部署与Pi Coding Agent集成指南

本文给出一套从环境核验、安装、模型下载、双卡启动，到Ubuntu非root安装Pi、注册本地模型和验证真实代码修改的完整流程。所有命令统一以CUDA 12.9为基准，不再混用其他CUDA变体。

> 本文面向当前这台8×H20服务器，Qwen默认使用GPU 0、1。模型放在2TB盘，Python环境、wheel、临时文件和日志放在700GB盘。已有DeepSeek配置保持不变。

## 1. 最终方案与验收标准

固定组合如下：

| 项目 | 固定值 |
| --- | --- |
| 系统CUDA Toolkit | 12.9 |
| nvcc | 12.9 |
| PyTorch | 2.13.0+cu129 |
| vLLM | 0.29.0+cu129 |
| vLLM wheel | vllm-0.29.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl |
| 模型 | Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 |
| GPU | 2×H20 |
| Qwen端口 | 8001 |
| 初始生产上下文 | 65,536 tokens |
| 工具调用解析器 | qwen3_xml |
| Pi Coding Agent | 0.86.1 |
| Node.js | ≥22.19.0 |
| Pi模型配置 | ~/.pi/agent/models.json |

最终必须同时满足：

~~~text
nvcc release 12.9
torch 2.13.0+cu129
torch.version.cuda 12.9
vLLM 0.29.0+cu129
vLLM动态库不依赖libcudart.so.13
Qwen API监听127.0.0.1:8001
Pi可通过qwen-local/qwen3-coder完成读取、运行、修改和复测
~~~

Qwen3-Coder-30B-A3B-Instruct-FP8原生支持262,144 tokens，但首次稳定运行建议从8K开始，确认双卡、API和工具调用均正常后，再提升到64K。128K和262K放在最后做阶梯测试。

## 2. 目录、变量与端口规划

磁盘规划：

~~~text
/mnt/tydrive/txhan/                         # 2TB盘
├── models/
│   └── Qwen3-Coder-30B-A3B-Instruct-FP8/
└── .cache/
    └── huggingface/

/mnt/data/txhan/                            # 700GB盘
├── .cache/
│   ├── pip/
│   ├── torch_extensions/
│   ├── triton/
│   ├── uv/
│   └── vllm/
└── qwen3-coder/
    ├── .venv/
    ├── logs/
    ├── tmp/
    ├── wheels/
    └── start_qwen.sh
~~~

Qwen只使用下面三个专属变量：

~~~bash
export QWEN_WORKDIR=/mnt/data/txhan/qwen3-coder
export QWEN_MODEL_DIR=/mnt/tydrive/txhan/models/Qwen3-Coder-30B-A3B-Instruct-FP8
export QWEN_PORT=8001
~~~

如果DeepSeek正在使用<code>MODEL_DIR</code>和8000端口，不要修改它们。推荐的最终端口分配是：

| 服务 | 端口 |
| --- | ---: |
| DeepSeek | 8000 |
| Qwen | 8001 |

检查两个模型目录没有混用：

~~~bash
printf 'DeepSeek MODEL_DIR=%s\n' "${MODEL_DIR:-<未设置>}"
printf 'Qwen QWEN_MODEL_DIR=%s\n' "$QWEN_MODEL_DIR"
~~~

## 3. 检查GPU、架构和CUDA 12.9

先检查服务器架构：

~~~bash
uname -m
~~~

固定wheel只适用于：

~~~text
x86_64
~~~

如果返回<code>aarch64</code>，不要继续使用本文的x86_64 wheel。

检查GPU及互联：

~~~bash
nvidia-smi
nvidia-smi topo -m
~~~

确认要使用的两张H20当前没有其他任务占用。本文默认：

~~~text
CUDA_VISIBLE_DEVICES=0,1
~~~

检查系统CUDA Toolkit：

~~~bash
command -v nvcc
readlink -f "$(command -v nvcc)"
nvcc --version
~~~

目标输出必须包含：

~~~text
release 12.9
~~~

注意：<code>nvidia-smi</code>顶部显示的CUDA版本是驱动可支持的最高CUDA版本，不等于当前shell实际使用的Toolkit版本；本指南以<code>nvcc --version</code>和<code>torch.version.cuda</code>为准。

如果CUDA 12.9已经安装在<code>/usr/local/cuda-12.9</code>，但当前shell调用了别的<code>nvcc</code>，只在Qwen会话中切换，不修改全局软链接：

~~~bash
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
hash -r

command -v nvcc
nvcc --version
~~~

## 4. 创建Qwen专属目录和缓存

~~~bash
export QWEN_WORKDIR=/mnt/data/txhan/qwen3-coder
export QWEN_MODEL_DIR=/mnt/tydrive/txhan/models/Qwen3-Coder-30B-A3B-Instruct-FP8
export QWEN_PORT=8001

mkdir -p "$QWEN_WORKDIR"/{logs,tmp,wheels}
mkdir -p "$QWEN_MODEL_DIR"
mkdir -p /mnt/data/txhan/.cache/{uv,pip,vllm,torch_extensions,triton}
mkdir -p /mnt/tydrive/txhan/.cache/huggingface
~~~

将各类缓存固定到数据盘：

~~~bash
export UV_CACHE_DIR=/mnt/data/txhan/.cache/uv
export PIP_CACHE_DIR=/mnt/data/txhan/.cache/pip
export VLLM_CACHE_ROOT=/mnt/data/txhan/.cache/vllm
export TORCH_EXTENSIONS_DIR=/mnt/data/txhan/.cache/torch_extensions
export TRITON_CACHE_DIR=/mnt/data/txhan/.cache/triton
export HF_HOME=/mnt/tydrive/txhan/.cache/huggingface
export TMPDIR="$QWEN_WORKDIR/tmp"
~~~

检查：

~~~bash
printf '%s\n' \
  "QWEN_WORKDIR=$QWEN_WORKDIR" \
  "QWEN_MODEL_DIR=$QWEN_MODEL_DIR" \
  "QWEN_PORT=$QWEN_PORT" \
  "UV_CACHE_DIR=$UV_CACHE_DIR" \
  "HF_HOME=$HF_HOME" \
  "TMPDIR=$TMPDIR"
~~~

## 5. 准备uv和Python 3.10虚拟环境

先检查uv：

~~~bash
command -v uv
uv --version
~~~

如果尚未安装uv，可执行官方安装命令：

~~~bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
~~~

进入工作目录：

~~~bash
cd /mnt/data/txhan/qwen3-coder
~~~

### 5.1 已修复环境

如果<code>.venv</code>已经修复成功，不要删除或重建，直接激活并进入第7节验证：

~~~bash
source .venv/bin/activate
python --version
~~~

### 5.2 全新安装

仅当<code>.venv</code>不存在时创建：

~~~bash
uv venv --python 3.10 .venv
source .venv/bin/activate

python --version
which python
~~~

<code>which python</code>应指向：

~~~text
/mnt/data/txhan/qwen3-coder/.venv/bin/python
~~~

## 6. 安装固定的cu129 vLLM wheel

不要让包管理器自动选择vLLM变体。固定使用v0.29.0官方release中的cu129 x86_64 wheel。

### 6.1 下载wheel

~~~bash
export QWEN_WORKDIR=/mnt/data/txhan/qwen3-coder
export VLLM_WHEEL="$QWEN_WORKDIR/wheels/vllm-0.29.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"

mkdir -p "$QWEN_WORKDIR/wheels"

wget -c -O "$VLLM_WHEEL" \
  'https://github.com/vllm-project/vllm/releases/download/v0.29.0/vllm-0.29.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl'

ls -lh "$VLLM_WHEEL"
~~~

官方release asset大小为548,493,389 bytes。下载后校验SHA-256：

~~~bash
printf '%s  %s\n' \
  '22e8d8fec755986b3ad964004a1f8c65a55626ec948354bdbe95993e6b0289fe' \
  "$VLLM_WHEEL" |
sha256sum -c -
~~~

结果应为<code>OK</code>。下载失败或校验不通过时，重新执行同一条<code>wget -c</code>命令即可断点续传，不要改成自动发现URL。

### 6.2 先安装新版packaging

~~~bash
cd /mnt/data/txhan/qwen3-coder
source .venv/bin/activate

uv pip install \
  --default-index https://pypi.org/simple \
  'packaging>=24.2'

python -c "import packaging; print(packaging.__version__)"
~~~

只要版本不低于24.2即可。

### 6.3 安装本地vLLM wheel

~~~bash
uv pip install "$VLLM_WHEEL" \
  --default-index https://pypi.org/simple \
  --index https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match
~~~

这里三项各有明确职责：

~~~text
本地wheel
  └─ 固定vLLM为0.29.0+cu129

官方PyPI
  └─ 提供packaging、flashinfer-python等普通依赖

PyTorch cu129索引
  └─ 提供torch和CUDA 12.9对应的Python依赖
~~~

<code>uv</code>默认的<code>first-index</code>策略可能在优先索引中找到包名后停止继续搜索，进而只看到过旧的<code>packaging</code>。因此这里必须显式使用：

~~~text
--index-strategy unsafe-best-match
~~~

本方案只组合官方PyPI和PyTorch官方cu129索引，不要再加入来源不明的第三方索引。

### 6.4 如果仍出现packaging依赖冲突

先显式安装普通依赖：

~~~bash
uv pip install \
  --default-index https://pypi.org/simple \
  'packaging>=24.2' \
  'flashinfer-python==0.6.18'
~~~

再重复本地wheel安装命令：

~~~bash
uv pip install "$VLLM_WHEEL" \
  --default-index https://pypi.org/simple \
  --index https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match
~~~

不需要因此删除<code>.venv</code>，也不需要改动系统CUDA Toolkit。

## 7. 安装后完整验证

### 7.1 检查依赖一致性

~~~bash
cd /mnt/data/txhan/qwen3-coder
source .venv/bin/activate

uv pip check
uv pip list | grep -Ei 'vllm|torch|flashinfer|packaging'
~~~

目标组合：

~~~text
packaging           24.2或更高
flashinfer-python   0.6.18
torch               2.13.0+cu129
vllm                0.29.0+cu129
~~~

### 7.2 在启动服务前执行导入测试

~~~bash
python - <<'PY'
import packaging
import torch
import vllm

print("packaging:", packaging.__version__)
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("vLLM:", vllm.__version__)

assert tuple(map(int, packaging.__version__.split(".")[:2])) >= (24, 2)
assert torch.__version__ == "2.13.0+cu129"
assert torch.version.cuda == "12.9"
assert torch.cuda.is_available()
assert vllm.__version__ == "0.29.0+cu129"

print("CUDA 12.9 STACK VERIFIED")
PY
~~~

### 7.3 检查vLLM动态库链接

~~~bash
VLLM_DIR="$(python -c 'import os, vllm; print(os.path.dirname(vllm.__file__))')"

find "$VLLM_DIR" -type f -name '*.so' -print0 |
while IFS= read -r -d '' f; do
    deps="$(readelf -d "$f" 2>/dev/null | grep libcudart || true)"
    if [ -n "$deps" ]; then
        echo "=== $f ==="
        echo "$deps"
    fi
done
~~~

cu129 wheel应链接CUDA 12系列运行库，例如：

~~~text
libcudart.so.12
~~~

不应出现：

~~~text
libcudart.so.13
~~~

如果导入成功、版本断言通过且没有CUDA 13运行库依赖，Python环境修复完成。

## 8. 下载Qwen FP8模型

模型固定为：

~~~text
Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8
~~~

先确认Hugging Face CLI可用：

~~~bash
cd /mnt/data/txhan/qwen3-coder
source .venv/bin/activate

command -v hf
hf --help
~~~

如果没有<code>hf</code>命令：

~~~bash
uv pip install \
  --default-index https://pypi.org/simple \
  'huggingface_hub[cli]'
~~~

下载到2TB盘：

~~~bash
export QWEN_MODEL_DIR=/mnt/tydrive/txhan/models/Qwen3-Coder-30B-A3B-Instruct-FP8
export HF_HOME=/mnt/tydrive/txhan/.cache/huggingface

hf download \
  Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
  --local-dir "$QWEN_MODEL_DIR"
~~~

如果下载中断，重复执行同一条命令。Hugging Face会复用已经完成的文件，不要删除整个模型目录。

### 8.1 验证模型文件

~~~bash
du -sh "$QWEN_MODEL_DIR"
ls -lh "$QWEN_MODEL_DIR"

test -f "$QWEN_MODEL_DIR/config.json"
test -f "$QWEN_MODEL_DIR/tokenizer.json"
test -f "$QWEN_MODEL_DIR/tokenizer_config.json"
test -f "$QWEN_MODEL_DIR/model.safetensors.index.json"

find "$QWEN_MODEL_DIR" -maxdepth 1 \
  -type f -name '*.safetensors' -printf '%f %s bytes\n' |
sort
~~~

查看模型配置中的原生上下文长度和量化方式：

~~~bash
python - <<'PY'
import json
from pathlib import Path

model_dir = Path("/mnt/tydrive/txhan/models/Qwen3-Coder-30B-A3B-Instruct-FP8")
config = json.loads((model_dir / "config.json").read_text())

print("model_type:", config.get("model_type"))
print("max_position_embeddings:", config.get("max_position_embeddings"))
print("quant_method:", config.get("quantization_config", {}).get("quant_method"))
print("weight_block_size:", config.get("quantization_config", {}).get("weight_block_size"))
PY
~~~

目标结果包括：

~~~text
model_type: qwen3_moe
max_position_embeddings: 262144
quant_method: fp8
weight_block_size: [128, 128]
~~~

## 9. 单卡8K最小启动测试

先用单卡和8K上下文排除模型文件、vLLM和基础CUDA问题：

~~~bash
cd /mnt/data/txhan/qwen3-coder
source .venv/bin/activate

export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/mnt/tydrive/txhan/.cache/huggingface
export TMPDIR=/mnt/data/txhan/qwen3-coder/tmp
export VLLM_CACHE_ROOT=/mnt/data/txhan/.cache/vllm
export TORCH_EXTENSIONS_DIR=/mnt/data/txhan/.cache/torch_extensions
export TRITON_CACHE_DIR=/mnt/data/txhan/.cache/triton

vllm serve /mnt/tydrive/txhan/models/Qwen3-Coder-30B-A3B-Instruct-FP8 \
  --served-model-name qwen3-coder \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --max-num-seqs 2 \
  --gpu-memory-utilization 0.85 \
  --host 127.0.0.1 \
  --port 8001
~~~

在另一个终端检查：

~~~bash
watch -n 1 nvidia-smi
~~~

看到服务完成模型加载后，再执行API测试。单卡测试通过后按<code>Ctrl+C</code>停止服务。

## 10. 双卡8K启动测试

~~~bash
cd /mnt/data/txhan/qwen3-coder
source .venv/bin/activate

export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export CUDA_VISIBLE_DEVICES=0,1
export HF_HOME=/mnt/tydrive/txhan/.cache/huggingface
export TMPDIR=/mnt/data/txhan/qwen3-coder/tmp
export VLLM_CACHE_ROOT=/mnt/data/txhan/.cache/vllm
export TORCH_EXTENSIONS_DIR=/mnt/data/txhan/.cache/torch_extensions
export TRITON_CACHE_DIR=/mnt/data/txhan/.cache/triton

vllm serve /mnt/tydrive/txhan/models/Qwen3-Coder-30B-A3B-Instruct-FP8 \
  --served-model-name qwen3-coder \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.85 \
  --host 127.0.0.1 \
  --port 8001
~~~

验证两个进程都使用到了GPU：

~~~bash
nvidia-smi
curl -fsS http://127.0.0.1:8001/v1/models
~~~

双卡8K通过后按<code>Ctrl+C</code>停止服务，再进入64K配置。

## 11. 双卡64K稳定配置

64K作为首个常驻配置：

~~~bash
cd /mnt/data/txhan/qwen3-coder
source .venv/bin/activate

export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export CUDA_VISIBLE_DEVICES=0,1
export HF_HOME=/mnt/tydrive/txhan/.cache/huggingface
export TMPDIR=/mnt/data/txhan/qwen3-coder/tmp
export VLLM_CACHE_ROOT=/mnt/data/txhan/.cache/vllm
export TORCH_EXTENSIONS_DIR=/mnt/data/txhan/.cache/torch_extensions
export TRITON_CACHE_DIR=/mnt/data/txhan/.cache/triton

vllm serve /mnt/tydrive/txhan/models/Qwen3-Coder-30B-A3B-Instruct-FP8 \
  --served-model-name qwen3-coder \
  --tensor-parallel-size 2 \
  --max-model-len 65536 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --host 127.0.0.1 \
  --port 8001
~~~

先不要同时修改多个性能参数。观察显存、首token延迟和并发稳定性后，再逐项调整<code>max-num-seqs</code>或<code>max-num-batched-tokens</code>。

## 12. 启用Qwen3-Coder工具调用

Qwen3-Coder使用vLLM的<code>qwen3_xml</code>工具调用解析器。启动参数增加：

~~~text
--enable-auto-tool-choice
--tool-call-parser qwen3_xml
~~~

完整命令：

~~~bash
vllm serve /mnt/tydrive/txhan/models/Qwen3-Coder-30B-A3B-Instruct-FP8 \
  --served-model-name qwen3-coder \
  --tensor-parallel-size 2 \
  --max-model-len 65536 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --host 127.0.0.1 \
  --port 8001
~~~

该模型只支持non-thinking模式，不需要添加reasoning parser，也不需要传<code>enable_thinking=False</code>。

## 13. 创建最终启动脚本

创建文件：

~~~bash
nano /mnt/data/txhan/qwen3-coder/start_qwen.sh
~~~

写入：

~~~bash
#!/usr/bin/env bash
set -euo pipefail

QWEN_WORKDIR=/mnt/data/txhan/qwen3-coder
QWEN_MODEL_DIR=/mnt/tydrive/txhan/models/Qwen3-Coder-30B-A3B-Instruct-FP8
QWEN_PORT=8001

CUDA_HOME=/usr/local/cuda-12.9
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export CUDA_VISIBLE_DEVICES=0,1

export UV_CACHE_DIR=/mnt/data/txhan/.cache/uv
export PIP_CACHE_DIR=/mnt/data/txhan/.cache/pip
export VLLM_CACHE_ROOT=/mnt/data/txhan/.cache/vllm
export TORCH_EXTENSIONS_DIR=/mnt/data/txhan/.cache/torch_extensions
export TRITON_CACHE_DIR=/mnt/data/txhan/.cache/triton
export HF_HOME=/mnt/tydrive/txhan/.cache/huggingface
export TMPDIR="$QWEN_WORKDIR/tmp"

mkdir -p \
  "$QWEN_WORKDIR/logs" \
  "$TMPDIR" \
  "$UV_CACHE_DIR" \
  "$PIP_CACHE_DIR" \
  "$VLLM_CACHE_ROOT" \
  "$TORCH_EXTENSIONS_DIR" \
  "$TRITON_CACHE_DIR" \
  "$HF_HOME"

source "$QWEN_WORKDIR/.venv/bin/activate"

python - <<'PY'
import torch
import vllm

assert torch.__version__ == "2.13.0+cu129", torch.__version__
assert torch.version.cuda == "12.9", torch.version.cuda
assert vllm.__version__ == "0.29.0+cu129", vllm.__version__

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("vLLM:", vllm.__version__)
PY

exec vllm serve "$QWEN_MODEL_DIR" \
  --served-model-name qwen3-coder \
  --tensor-parallel-size 2 \
  --max-model-len 65536 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --host 127.0.0.1 \
  --port "$QWEN_PORT"
~~~

保存后加执行权限：

~~~bash
chmod +x /mnt/data/txhan/qwen3-coder/start_qwen.sh
~~~

检查脚本语法：

~~~bash
bash -n /mnt/data/txhan/qwen3-coder/start_qwen.sh
~~~

启动：

~~~bash
/mnt/data/txhan/qwen3-coder/start_qwen.sh
~~~

这个脚本没有读取或修改DeepSeek使用的<code>MODEL_DIR</code>。

## 14. API验证

### 14.1 模型列表

~~~bash
curl -fsS http://127.0.0.1:8001/v1/models
~~~

返回结果中应包含：

~~~text
qwen3-coder
~~~

### 14.2 普通对话

~~~bash
curl -fsS http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-coder",
    "messages": [
      {
        "role": "user",
        "content": "Write a Python function that returns the nth Fibonacci number and explain its complexity."
      }
    ],
    "temperature": 0.7,
    "top_p": 0.8,
    "max_tokens": 1024
  }'
~~~

### 14.3 工具调用

~~~bash
curl -fsS http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-coder",
    "messages": [
      {
        "role": "user",
        "content": "Use the calculator tool to multiply 37 by 19."
      }
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "calculator",
          "description": "Evaluate a multiplication expression.",
          "parameters": {
            "type": "object",
            "properties": {
              "a": {"type": "number"},
              "b": {"type": "number"}
            },
            "required": ["a", "b"]
          }
        }
      }
    ],
    "tool_choice": "auto",
    "temperature": 0,
    "max_tokens": 512
  }'
~~~

目标是响应中的<code>choices[0].message.tool_calls</code>包含<code>calculator</code>及参数37、19。vLLM只负责生成工具调用；真正执行函数、把结果追加为tool消息并再次请求模型，是调用端的职责。

## 15. 在Ubuntu中以非root用户安装Pi

Pi和Qwen应运行在同一台Ubuntu服务器上。Qwen占用GPU 0、1并监听本机8001端口；Pi使用当前普通用户的权限访问项目文件，不需要也不应以root身份启动。

先确认当前身份：

~~~bash
whoami
id -u
echo "$HOME"
~~~

预期用户名类似<code>txhan</code>，UID不应为0，HOME应类似<code>/home/txhan</code>。如果当前就是root，请先切换回日常使用的普通用户。

### 15.1 检查Node.js

Pi 0.86.1要求Node.js 22.19.0或更高版本：

~~~bash
node -v
npm -v
~~~

如果Node不存在或版本过低，推荐通过nvm安装。nvm安装在当前用户目录，不会把Node和npm包写入<code>/root</code>或系统目录：

~~~bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.6/install.sh | bash
source ~/.bashrc

nvm install 22
nvm use 22
nvm alias default 22

node -v
npm -v
command -v node
command -v npm
~~~

确认Node版本不低于22.19.0，并且路径位于当前用户的<code>~/.nvm</code>目录。

如果必须继续使用系统Node，可把npm全局目录改到当前用户HOME：

~~~bash
mkdir -p ~/.local/npm
npm config set prefix ~/.local/npm
grep -qxF 'export PATH="$HOME/.local/npm/bin:$PATH"' ~/.bashrc \
  || echo 'export PATH="$HOME/.local/npm/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

npm config get prefix
~~~

使用nvm时通常不需要这组npm prefix命令。

### 15.2 安装固定版本的Pi

~~~bash
npm install -g --ignore-scripts @earendil-works/pi-coding-agent@0.86.1

pi --version
command -v pi
~~~

先确认Pi本身能够进入交互界面：

~~~bash
pi
~~~

完成检查后按<code>Ctrl+C</code>退出。不要使用<code>sudo npm install -g</code>，也不要使用<code>sudo pi</code>。

## 16. 启动Qwen并确认工具调用模式

Pi接入前，先启动第13节生成的固定脚本。推荐在tmux中运行：

~~~bash
tmux new -s qwen

/mnt/data/txhan/qwen3-coder/start_qwen.sh \
  2>&1 | tee -a /mnt/data/txhan/qwen3-coder/logs/qwen.log
~~~

看到服务完成加载后，按<code>Ctrl+B</code>，再按<code>D</code>退出tmux但保持服务运行。

在另一个Ubuntu终端验证：

~~~bash
curl -fsS http://127.0.0.1:8001/v1/models
ss -ltnp | grep ':8001'
~~~

返回模型列表中必须包含<code>qwen3-coder</code>。启动日志或进程参数中还必须同时包含：

~~~text
--enable-auto-tool-choice
--tool-call-parser qwen3_xml
~~~

如果第14.3节的原始API工具调用测试尚未通过，先修复vLLM侧问题，不要继续配置Pi。

## 17. 备份并配置Pi的本地Qwen模型

Pi在Ubuntu中的自定义模型配置文件是：

~~~text
~/.pi/agent/models.json
~~~

先创建目录，并对已有配置做时间戳备份：

~~~bash
mkdir -p ~/.pi/agent

if [ -f ~/.pi/agent/models.json ]; then
  cp -a ~/.pi/agent/models.json \
    ~/.pi/agent/models.json.bak.$(date +%Y%m%d-%H%M%S)
fi
~~~

如果文件不存在，新建<code>~/.pi/agent/models.json</code>并写入：

~~~json
{
  "providers": {
    "qwen-local": {
      "baseUrl": "http://127.0.0.1:8001/v1",
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
~~~

如果文件中已经有其他provider，只把<code>qwen-local</code>对象合并到现有<code>providers</code>中，不要整文件覆盖。<code>apiKey</code>使用<code>local</code>是有意设置的dummy key；本机vLLM虽然不校验密钥，但Pi需要该字段来正常启用模型。

配置中的<code>contextWindow</code>必须与vLLM的<code>--max-model-len</code>一致。本文首个常驻配置统一为65,536。

## 18. 验证JSON、模型列表和基础连通性

先验证JSON语法。以下两种方法任选一种：

~~~bash
jq . ~/.pi/agent/models.json
~~~

或：

~~~bash
python3 -m json.tool ~/.pi/agent/models.json
~~~

然后确认API仍在线，并让Pi加载自定义模型：

~~~bash
curl -fsS http://127.0.0.1:8001/v1/models
pi --list-models qwen
~~~

模型列表中应出现：

~~~text
qwen-local/qwen3-coder
~~~

第一次先用one-shot请求验证完整链路：

~~~bash
pi --model qwen-local/qwen3-coder \
  -p "Reply with exactly LOCAL_QWEN_OK"
~~~

目标输出为：

~~~text
LOCAL_QWEN_OK
~~~

这一步只证明<code>Pi → models.json → vLLM → Qwen</code>的普通请求已连通，还不能替代后面的工具调用测试。

## 19. 在安全目录中验证读取和工具调用

先使用专门的测试目录，不要直接让模型操作重要仓库：

~~~bash
mkdir -p ~/pi-qwen-test
cd ~/pi-qwen-test
printf 'hello\n' > test.txt

pi --model qwen-local/qwen3-coder
~~~

在Pi中输入：

~~~text
Use the bash tool to run pwd and ls -la. Then read test.txt and tell me its contents. Do not modify anything.
~~~

预期过程是Qwen生成结构化工具调用，Pi依次执行<code>bash</code>、<code>ls</code>或<code>read</code>，再把结果返回给Qwen。最终应正确报告<code>test.txt</code>内容为<code>hello</code>，并且文件没有变化。

Pi内置的常用工具包括<code>read</code>、<code>bash</code>、<code>edit</code>、<code>write</code>、<code>grep</code>、<code>find</code>和<code>ls</code>。真正访问文件和执行命令的是Pi进程，不是vLLM服务本身。

## 20. 验证文件修改

仍在<code>~/pi-qwen-test</code>中，让Pi执行：

~~~text
Change test.txt from "hello" to "hello from local Qwen". Then read it back to verify the change.
~~~

退出Pi后验证：

~~~bash
cat ~/pi-qwen-test/test.txt
~~~

预期结果：

~~~text
hello from local Qwen
~~~

如果模型只输出建议而没有调用<code>edit</code>或<code>write</code>，先检查第16节的两个工具调用参数和第14.3节的原始API响应，不要通过提升Pi权限来规避。

## 21. 验证真实代码定位、修复和复测

创建一个可安全破坏的最小Python项目：

~~~bash
mkdir -p ~/pi-code-test
cd ~/pi-code-test

cat > calc.py <<'PY'
def add(a, b):
    return a - b

print(add(2, 3))
PY

pi --model qwen-local/qwen3-coder
~~~

在Pi中输入：

~~~text
Inspect the current directory.
Run calc.py.
Find the bug.
Fix it.
Run it again and verify the output.
~~~

完整链路应类似：

~~~text
Qwen生成tool_calls
  ↓
Pi执行ls和read
  ↓
Pi执行python calc.py，首次得到-1
  ↓
Qwen请求edit，把减法改为加法
  ↓
Pi再次执行python calc.py
  ↓
输出5
~~~

退出Pi后再独立复核：

~~~bash
cd ~/pi-code-test
python3 calc.py
sed -n '1,20p' calc.py
~~~

只有读取、运行、修改和复测全部完成，才说明Pi与本地Qwen的Coding Agent链路真正跑通。

## 22. 保存默认模型和可选快捷命令

### 22.1 在Pi中保存默认模型

启动Pi：

~~~bash
pi
~~~

输入：

~~~text
/model
~~~

选择<code>qwen-local/qwen3-coder</code>后，按<code>Ctrl+S</code>保存为默认模型。只选中模型而不按<code>Ctrl+S</code>，不会持久化默认设置。

Pi保存的对应设置字段是<code>defaultProvider</code>、<code>defaultModel</code>和<code>defaultThinkingLevel</code>。保存后退出并重新执行<code>pi</code>，确认默认模型仍是本地Qwen。

### 22.2 创建固定模型的包装脚本

如果不想依赖默认设置，可创建一个显式指定模型的命令：

~~~bash
mkdir -p ~/.local/bin

cat > ~/.local/bin/pi-qwen <<'SH'
#!/usr/bin/env bash
exec pi --model qwen-local/qwen3-coder "$@"
SH

chmod +x ~/.local/bin/pi-qwen

grep -qxF 'export PATH="$HOME/.local/bin:$PATH"' ~/.bashrc \
  || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
~~~

以后进入任意项目后执行：

~~~bash
cd /你的项目
pi-qwen
~~~

## 23. tmux常驻与日常使用

如果系统没有tmux，并且当前用户有管理员权限：

~~~bash
sudo apt update
sudo apt install -y tmux
~~~

日常启动Qwen：

~~~bash
tmux new -s qwen

/mnt/data/txhan/qwen3-coder/start_qwen.sh \
  2>&1 | tee -a /mnt/data/txhan/qwen3-coder/logs/qwen.log
~~~

按<code>Ctrl+B</code>，再按<code>D</code>脱离会话。常用维护命令：

~~~bash
tmux ls
tmux attach -t qwen
tail -f /mnt/data/txhan/qwen3-coder/logs/qwen.log
~~~

停止服务时，重新进入tmux会话并按<code>Ctrl+C</code>，让vLLM正常退出。

确认API在线后，进入实际代码仓库：

~~~bash
cd /你的代码项目
pi --model qwen-local/qwen3-coder
~~~

第一次处理真实仓库时，建议先给只读指令：

~~~text
Inspect this repository and explain its architecture.
Do not modify anything yet.
~~~

确认模型理解正确后，再让它运行测试和修改文件。

## 24. 非root权限与安全习惯

Pi通过当前Ubuntu用户实际执行<code>bash</code>、<code>read</code>、<code>edit</code>和<code>write</code>。因此：

- 不要运行<code>sudo pi</code>；
- 不要把当前用户无条件加入高权限组；
- 首次测试只使用<code>~/pi-qwen-test</code>和<code>~/pi-code-test</code>；
- 进入重要仓库前先提交或备份现有改动；
- 先让Pi解释将要执行的高风险命令，再由人工决定是否执行；
- 对<code>apt</code>、<code>systemctl</code>、<code>/etc</code>、驱动和CUDA系统目录的修改保留人工确认；
- 不向公网直接开放无鉴权的8001端口。

Qwen本身不直接获得Linux权限。它只生成结构化<code>tool_calls</code>；vLLM通过<code>qwen3_xml</code>转换这些调用，Pi再以当前用户身份执行工具。

## 25. Windows通过SSH隧道访问（可选）

Pi与Qwen都运行在Ubuntu服务器时不需要SSH隧道，直接使用<code>127.0.0.1:8001</code>即可。

如果要从Windows访问服务器上的Qwen，在Windows PowerShell中执行：

~~~powershell
ssh -N -L 8001:127.0.0.1:8001 用户名@服务器IP
~~~

保持该窗口运行，再在另一个PowerShell窗口验证：

~~~powershell
curl.exe http://127.0.0.1:8001/v1/models
~~~

如果还要在Windows上运行Pi，Windows侧也需要单独安装Pi，并在Windows用户目录的<code>.pi\agent\models.json</code>中配置同一个<code>qwen-local</code>provider。不要把Ubuntu的<code>~/.pi</code>路径照搬成Windows路径。

不要为了省去隧道而把vLLM改成<code>--host 0.0.0.0</code>并直接暴露到公网。

## 26. 从64K逐步提升上下文

模型原生上下文长度为262,144，但模型支持不等于当前显存和并发配置一定稳定。每次只调整一档，并同步修改vLLM与Pi。

### 26.1 128K

将<code>start_qwen.sh</code>改为：

~~~text
--max-model-len 131072
--max-num-seqs 2
~~~

同时把<code>models.json</code>改为：

~~~json
"contextWindow": 131072
~~~

验证两张GPU都没有OOM、连续工具调用不报错、首token延迟可接受后，再继续下一档。

### 26.2 原生262K

128K稳定后再测试：

~~~text
--max-model-len 262144
--max-num-seqs 1
~~~

Pi配置同步改为：

~~~json
"contextWindow": 262144
~~~

如果OOM、吞吐明显下降或长工具链不稳定，退回上一档。不要把Pi的<code>contextWindow</code>写得高于vLLM的<code>--max-model-len</code>，也不要通过填写大于262,144的数值来冒充更长上下文。

## 27. Qwen与Pi联合故障排查

### 27.1 找不到pi命令

~~~bash
command -v node
command -v npm
command -v pi
npm config get prefix
~~~

如果使用nvm，重新执行<code>source ~/.bashrc</code>和<code>nvm use 22</code>。如果使用用户npm prefix，确认<code>~/.local/npm/bin</code>已加入PATH。不要改用sudo安装来掩盖PATH问题。

### 27.2 Pi模型列表中没有qwen-local

依次检查：

~~~bash
python3 -m json.tool ~/.pi/agent/models.json
pi --list-models qwen
~~~

确认provider名称为<code>qwen-local</code>、模型ID为<code>qwen3-coder</code>、<code>apiKey</code>存在且值为<code>local</code>，并确认配置文件属于当前用户而不是root。

### 27.3 Pi提示连接失败或Connection refused

~~~bash
curl -fsS http://127.0.0.1:8001/v1/models
ss -ltnp | grep ':8001'
tmux ls
tail -n 100 /mnt/data/txhan/qwen3-coder/logs/qwen.log
~~~

如果curl失败，问题在Qwen服务或端口，不在Pi配置。确认Qwen使用8001，不要误连DeepSeek的8000端口。

### 27.4 普通对话成功，但工具调用失败

先重新执行第14.3节的原始API工具调用测试。检查Qwen启动命令是否同时包含：

~~~text
--enable-auto-tool-choice
--tool-call-parser qwen3_xml
~~~

不要添加reasoning parser。若模型把工具XML当普通文本输出，重点检查vLLM版本、模型名、tool parser参数和启动日志。

### 27.5 再次出现libcudart.so.13

说明当前环境中的vLLM不是本文固定的cu129 wheel，或有错误动态库被优先加载：

~~~bash
source /mnt/data/txhan/qwen3-coder/.venv/bin/activate

python -c "import torch, vllm; print(torch.__version__, torch.version.cuda, vllm.__version__)"
echo "$LD_LIBRARY_PATH"
~~~

重新安装固定wheel：

~~~bash
export VLLM_WHEEL=/mnt/data/txhan/qwen3-coder/wheels/vllm-0.29.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl

uv pip uninstall vllm

uv pip install "$VLLM_WHEEL" \
  --default-index https://pypi.org/simple \
  --index https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match
~~~

随后重新执行第7节全部验证。

### 27.6 出现packaging版本冲突

~~~bash
uv pip install \
  --default-index https://pypi.org/simple \
  'packaging>=24.2' \
  'flashinfer-python==0.6.18'
~~~

再安装本地cu129 wheel，并保留<code>--index-strategy unsafe-best-match</code>。

### 27.7 nvcc不是12.9

~~~bash
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
hash -r

command -v nvcc
nvcc --version
~~~

不要为了Qwen修改DeepSeek正在使用的全局CUDA配置；CUDA路径应固定在Qwen的<code>start_qwen.sh</code>中。

### 27.8 双卡启动失败、NCCL报错或OOM

~~~bash
nvidia-smi
nvidia-smi topo -m
~~~

临时增加诊断日志：

~~~bash
export NCCL_DEBUG=INFO
/mnt/data/txhan/qwen3-coder/start_qwen.sh
~~~

OOM时按以下顺序降低压力：

1. 将<code>max-model-len</code>从64K降到32K或8K；
2. 将<code>max-num-seqs</code>降到2或1；
3. 将<code>max-num-batched-tokens</code>降到4096；
4. 确认GPU 0、1没有其他进程；
5. 最后再小幅降低<code>gpu-memory-utilization</code>。

不要把<code>NCCL_P2P_DISABLE=1</code>之类的规避参数直接写入最终脚本。只有日志证明对应路径异常时，才做单项对照测试。

### 27.9 Pi读取或修改文件时Permission denied

~~~bash
whoami
pwd
ls -ld .
ls -l
~~~

Pi只能使用当前用户已有的文件权限。不要通过<code>sudo pi</code>解决；应先确认项目所有者和目标路径，再对单个必要目录做最小权限修复。

### 27.10 端口8001占用或模型下载不完整

检查占用者：

~~~bash
ss -ltnp | grep ':8001'
~~~

确认进程后再停止旧Qwen实例或调整Qwen专属端口，不要误停DeepSeek。

模型不完整时可重复断点下载：

~~~bash
hf download \
  Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
  --local-dir /mnt/tydrive/txhan/models/Qwen3-Coder-30B-A3B-Instruct-FP8
~~~

不要先删除整个模型目录。

## 28. 明确禁止的操作

不要执行未指定CUDA变体的vLLM升级或重装命令：

~~~text
uv pip install -U vllm
uv pip install vllm --torch-backend=auto
pip install -U vllm
~~~

也不要：

- 运行<code>sudo pi</code>或使用root的Pi配置；
- 使用<code>sudo npm install -g</code>安装Pi；
- 用自动脚本临时抓取“最新”vLLM wheel并覆盖固定版本；
- 把Qwen目录写回DeepSeek的<code>MODEL_DIR</code>；
- 把Qwen与DeepSeek放进同一个Python虚拟环境；
- 因单个依赖冲突就删除整个<code>.venv</code>；
- 未确认路径时递归删除模型、缓存或虚拟环境；
- 只看<code>nvidia-smi</code>就判断CUDA Toolkit版本；
- 在未完成8K、64K和Pi工具测试前直接上262K；
- 把无鉴权的vLLM接口直接暴露到公网；
- 在没有提交或备份的真实仓库中直接授权大范围修改。

需要升级时，应重新确定一组完整兼容矩阵，同时验证CUDA Toolkit、PyTorch、vLLM wheel、FlashInfer、动态库链接、Pi版本和模型配置，不能只升级其中一个组件。

## 29. 最终架构与职责边界

~~~text
Ubuntu Server
│
├── GPU 0 ─────┐
│              ├── Qwen3-Coder-30B-A3B-Instruct-FP8
├── GPU 1 ─────┘                  │
│                                ▼
│                         vLLM 0.29.0+cu129
│                         qwen3_xml parser
│                                │
│                     127.0.0.1:8001/v1
│                                │
│                                ▼
│                         Pi Coding Agent
│                                │
│              ┌─────────────────┼─────────────────┐
│              ▼                 ▼                 ▼
│            bash              read          edit / write
│              └─────────────────┼─────────────────┘
│                                ▼
│                         当前用户的代码仓库
│                         run / test / fix
│
└── GPU 2–7保留给DeepSeek或其他实验
~~~

职责边界是：

~~~text
Qwen只生成结构化tool_calls
        ↓
vLLM使用qwen3_xml完成协议转换
        ↓
Pi以当前Ubuntu用户身份调用本地工具
        ↓
工具结果返回Qwen，由模型决定下一步
~~~

因此，工具调用能否工作取决于Qwen、vLLM解析器、Pi配置和当前用户权限四层同时正确。

## 30. 最终检查清单

### 30.1 Qwen环境

~~~bash
nvcc --version

source /mnt/data/txhan/qwen3-coder/.venv/bin/activate

python - <<'PY'
import torch
import vllm
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("vLLM:", vllm.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
PY

uv pip check
~~~

应确认：

- <code>nvcc</code>是12.9；
- <code>torch</code>是2.13.0+cu129；
- <code>torch.version.cuda</code>是12.9；
- <code>vLLM</code>是0.29.0+cu129；
- vLLM动态库不依赖<code>libcudart.so.13</code>；
- 两张H20都在工作；
- Qwen只使用<code>QWEN_MODEL_DIR</code>、<code>QWEN_WORKDIR</code>和<code>QWEN_PORT</code>；
- DeepSeek的<code>MODEL_DIR</code>和8000端口未被修改。

### 30.2 服务与Pi

~~~bash
curl -fsS http://127.0.0.1:8001/v1/models
ss -ltnp | grep ':8001'
nvidia-smi

node -v
pi --version
python3 -m json.tool ~/.pi/agent/models.json
pi --list-models qwen
~~~

最终必须全部满足：

- Qwen监听127.0.0.1:8001；
- 普通对话和<code>qwen3_xml</code>工具调用都通过；
- Pi 0.86.1由普通Ubuntu用户安装和运行；
- Node.js不低于22.19.0；
- Pi能识别<code>qwen-local/qwen3-coder</code>；
- one-shot请求返回<code>LOCAL_QWEN_OK</code>；
- Pi能在安全目录中读取<code>test.txt</code>；
- Pi能修改文件并读回验证；
- Pi能运行、定位、修复并复测<code>calc.py</code>；
- 64K稳定后才继续测试128K或262K。

## 31. 官方参考资料

- [Pi Coding Agent仓库](https://github.com/earendil-works/pi)
- [Pi自定义模型与vLLM配置](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/models.md)
- [Pi Coding Agent README](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/README.md)
- [Pi设置说明](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/settings.md)
- [Pi Coding Agent npm包](https://www.npmjs.com/package/@earendil-works/pi-coding-agent)
- [nvm官方仓库](https://github.com/nvm-sh/nvm)
- [vLLM v0.29.0 Release](https://github.com/vllm-project/vllm/releases/tag/v0.29.0)
- [vLLM CUDA安装文档](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
- [vLLM Tool Calling文档](https://docs.vllm.ai/en/latest/features/tool_calling/)
- [Qwen3-Coder-30B-A3B-Instruct-FP8模型页](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8)
- [uv索引策略说明](https://docs.astral.sh/uv/reference/settings/#index-strategy)
- [PyTorch cu129索引](https://download.pytorch.org/whl/cu129)
