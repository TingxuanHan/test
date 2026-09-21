# Qwen3-Coder-30B-A3B-Instruct-FP8双H20部署指南

本文给出一套从环境核验、安装、模型下载、双卡启动，到OpenAI兼容API与Coding Agent接入的完整流程。所有命令统一以CUDA 12.9为基准，不再混用其他CUDA变体。

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

最终必须同时满足：

~~~text
nvcc release 12.9
torch 2.13.0+cu129
torch.version.cuda 12.9
vLLM 0.29.0+cu129
vLLM动态库不依赖libcudart.so.13
Qwen API监听127.0.0.1:8001
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

## 15. 使用tmux常驻运行

创建会话：

~~~bash
tmux new -s qwen
~~~

在会话中启动并记录日志：

~~~bash
/mnt/data/txhan/qwen3-coder/start_qwen.sh \
  2>&1 | tee -a /mnt/data/txhan/qwen3-coder/logs/qwen.log
~~~

按<code>Ctrl+B</code>，再按<code>D</code>退出tmux但保持服务运行。

查看会话：

~~~bash
tmux ls
~~~

重新进入：

~~~bash
tmux attach -t qwen
~~~

查看日志：

~~~bash
tail -f /mnt/data/txhan/qwen3-coder/logs/qwen.log
~~~

停止服务时进入tmux会话并按<code>Ctrl+C</code>，让vLLM正常退出。

## 16. Windows通过SSH隧道访问

服务默认只监听服务器本机的127.0.0.1，更安全。Windows PowerShell执行：

~~~powershell
ssh -N -L 8001:127.0.0.1:8001 用户名@服务器IP
~~~

保持该窗口运行，然后在另一个PowerShell窗口测试：

~~~powershell
curl.exe http://127.0.0.1:8001/v1/models
~~~

如果改用<code>--host 0.0.0.0</code>直接开放端口，应同时配置服务器防火墙、访问控制和API鉴权；不要把无鉴权接口直接暴露到公网。

## 17. 接入Pi

Windows配置文件：

~~~text
C:\Users\TxHan\.pi\agent\models.json
~~~

在现有<code>providers</code>中加入以下provider；如果文件里已有其他provider，不要整文件覆盖：

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

启动Pi：

~~~powershell
pi
~~~

执行：

~~~text
/model
~~~

选择：

~~~text
Qwen3-Coder 30B A3B FP8 - 2xH20
~~~

## 18. 进行真实Coding Agent测试

不要只测试简单问答。进入一个可安全测试的Git仓库：

~~~powershell
cd D:\你的项目
pi
~~~

先进行只读任务：

~~~text
Inspect this repository first.
Run the existing tests.
Identify the current failures and explain the likely cause.
Do not modify files yet.
~~~

完整链路应表现为：

~~~text
读取仓库
  ↓
发起工具调用
  ↓
调用端执行terminal命令
  ↓
把stdout/stderr作为tool结果返回
  ↓
模型继续分析
~~~

同时检查：

- Qwen日志中请求正常完成，没有tool parser错误；
- Pi能正确识别<code>tool_calls</code>；
- 模型没有把工具调用XML当普通文本输出；
- API响应模型名始终是<code>qwen3-coder</code>；
- 长任务不会因SSH隧道、客户端超时或上下文上限中断。

## 19. 从64K逐步提升上下文

模型原生上下文长度为262,144，但可支持不等于当前并发配置一定稳定。每次只改<code>max-model-len</code>和必要的并发参数，完成长提示测试后再进入下一档。

### 19.1 128K

将启动脚本改为：

~~~text
--max-model-len 131072
--max-num-seqs 2
~~~

其余参数先保持不变。验证：

- 两张GPU均无OOM；
- 能接受接近目标长度的真实代码仓库上下文；
- 首token延迟在可接受范围；
- 连续请求不会出现NCCL或CUDA错误。

Pi配置同步改为：

~~~json
"contextWindow": 131072
~~~

### 19.2 原生262K

128K稳定后再测试：

~~~text
--max-model-len 262144
--max-num-seqs 1
~~~

Pi配置同步改为：

~~~json
"contextWindow": 262144
~~~

如果OOM或吞吐明显不可接受，退回上一档。不要通过写入大于262,144的值来冒充更长上下文；超过原生长度需要额外的长上下文扩展方案，不属于本文的稳定基线。

## 20. 常见问题

### 20.1 再次出现libcudart.so.13

说明当前环境中的vLLM不是本文固定的cu129 wheel，或残留动态库被优先加载。

检查：

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

### 20.2 出现packaging版本冲突

~~~bash
uv pip install \
  --default-index https://pypi.org/simple \
  'packaging>=24.2' \
  'flashinfer-python==0.6.18'
~~~

再安装本地wheel，并保留<code>--index-strategy unsafe-best-match</code>。

### 20.3 nvcc不是12.9

~~~bash
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
hash -r

command -v nvcc
nvcc --version
~~~

不要为了Qwen修改DeepSeek正在使用的全局CUDA配置；优先把CUDA路径固定在<code>start_qwen.sh</code>内。

### 20.4 双卡启动失败或NCCL报错

先检查拓扑和GPU占用：

~~~bash
nvidia-smi
nvidia-smi topo -m
~~~

临时增加诊断日志：

~~~bash
export NCCL_DEBUG=INFO
/mnt/data/txhan/qwen3-coder/start_qwen.sh
~~~

不要把<code>NCCL_P2P_DISABLE=1</code>之类的规避参数直接写入最终脚本。只有在日志证明P2P路径异常时，才做单项对照测试。

### 20.5 OOM

按以下顺序降低资源压力：

1. 将<code>max-model-len</code>从64K降到32K或8K；
2. 将<code>max-num-seqs</code>降到2或1；
3. 将<code>max-num-batched-tokens</code>降到4096；
4. 确认GPU没有其他进程；
5. 最后再小幅降低<code>gpu-memory-utilization</code>。

模型官方也建议OOM时先把上下文长度降到32,768。

### 20.6 端口8001已占用

~~~bash
ss -ltnp | grep ':8001'
~~~

先确认占用者，再决定停止旧Qwen进程或调整Qwen专属端口。不要误停DeepSeek进程。

### 20.7 模型下载不完整

直接重复：

~~~bash
hf download \
  Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
  --local-dir /mnt/tydrive/txhan/models/Qwen3-Coder-30B-A3B-Instruct-FP8
~~~

不要先删除整个模型目录。

## 21. 明确禁止的操作

不要执行未指定CUDA变体的vLLM升级或重装命令，例如：

~~~text
uv pip install -U vllm
uv pip install vllm --torch-backend=auto
pip install -U vllm
~~~

也不要：

- 使用自动脚本临时抓取“最新”wheel并覆盖固定版本；
- 把Qwen目录写回DeepSeek的<code>MODEL_DIR</code>；
- 将Qwen与DeepSeek放进同一个Python虚拟环境；
- 因单个依赖冲突就删除整个<code>.venv</code>；
- 未确认路径时递归删除模型、缓存或虚拟环境；
- 只看<code>nvidia-smi</code>就判断Toolkit版本；
- 在未完成8K和64K验证前直接上262K生产配置。

需要升级时，应重新确定一组完整兼容矩阵，并同时验证系统Toolkit、PyTorch、vLLM wheel、FlashInfer和动态库链接，不能只升级其中一个包。

## 22. 最终检查清单

环境：

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

服务：

~~~bash
curl -fsS http://127.0.0.1:8001/v1/models
ss -ltnp | grep ':8001'
nvidia-smi
~~~

最终应确认：

- <code>nvcc</code>是12.9；
- <code>torch</code>是2.13.0+cu129；
- <code>torch.version.cuda</code>是12.9；
- <code>vLLM</code>是0.29.0+cu129；
- 两张H20都在工作；
- Qwen只使用<code>QWEN_MODEL_DIR</code>、<code>QWEN_WORKDIR</code>和<code>QWEN_PORT</code>；
- Qwen监听8001，DeepSeek配置未被修改；
- 普通对话和<code>qwen3_xml</code>工具调用都通过；
- 64K稳定后才继续测试128K或262K。

## 23. 官方参考资料

- [vLLM v0.29.0 Release](https://github.com/vllm-project/vllm/releases/tag/v0.29.0)
- [vLLM CUDA安装文档](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
- [vLLM Tool Calling文档](https://docs.vllm.ai/en/latest/features/tool_calling/)
- [Qwen3-Coder-30B-A3B-Instruct-FP8模型页](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8)
- [uv索引策略说明](https://docs.astral.sh/uv/reference/settings/#index-strategy)
- [PyTorch cu129索引](https://download.pytorch.org/whl/cu129)
