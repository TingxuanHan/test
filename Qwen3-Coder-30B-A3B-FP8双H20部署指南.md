# Qwen3-Coder-30B-A3B-Instruct-FP8 双 H20 部署指南

## 部署规划

建议按以下目录规划：**模型放在 2TB 盘，运行环境和日志放在 700GB 盘**。

```text
/mnt/tydrive/txhan/
├── models/
│   └── Qwen3-Coder-30B-A3B-Instruct-FP8/   # 模型，约31.2GB
└── .cache/huggingface/                      # HF缓存

/mnt/data/txhan/
└── qwen3-coder/
    ├── .venv/                                # Python/vLLM环境
    ├── logs/
    ├── tmp/
    └── start_qwen.sh
```

这次我建议你先用 **Qwen3-Coder-30B-A3B-Instruct-FP8**，而不是 BF16。它是 Qwen 官方 FP8 checkpoint，完整仓库约 **31.2GB**，官方明确支持直接用 vLLM；你的 H20 96GB 属于 Hopper，FP8 很合适。([Hugging Face][1])

另外，这个 Coder 版本是 **non-thinking 模型**，所以启动时**不要加 reasoning parser**；它有 30.5B 总参数、3.3B 激活参数、原生 262K context。([Hugging Face][2])

---

## 1. 检查磁盘与 GPU

先执行：

```bash
df -h /mnt/data/txhan /mnt/tydrive/txhan
```

然后：

```bash
nvidia-smi
```

建议再看一下 8 卡拓扑：

```bash
nvidia-smi topo -m
```

你主要观察 GPU 0 和 GPU 1 之间是什么连接。

如果是类似：

```text
NV1
NV2
NV4
NV8
```

说明有 NVLink/NVSwitch，适合 TP=2。

如果 GPU 0/1 正在被别人用，先看：

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv
```

然后选择两个空闲 GPU。

下面我先假设使用：

```text
GPU 0
GPU 1
```

---

## 2. 建目录

执行：

```bash
mkdir -p /mnt/tydrive/txhan/models
mkdir -p /mnt/tydrive/txhan/.cache/huggingface

mkdir -p /mnt/data/txhan/qwen3-coder
mkdir -p /mnt/data/txhan/qwen3-coder/logs
mkdir -p /mnt/data/txhan/qwen3-coder/tmp
```

进入工作目录：

```bash
cd /mnt/data/txhan/qwen3-coder
```

建议以后统一：

```bash
export HF_HOME=/mnt/tydrive/txhan/.cache/huggingface
export TMPDIR=/mnt/data/txhan/qwen3-coder/tmp
```

检查：

```bash
echo $HF_HOME
echo $TMPDIR
```

应该分别显示：

```text
/mnt/tydrive/txhan/.cache/huggingface
/mnt/data/txhan/qwen3-coder/tmp
```

---

## 3. 创建独立 vLLM 环境

你之前这台机器已经在用 `uv`，这里继续用它最省事。vLLM 当前官方也推荐：

```bash
uv pip install vllm --torch-backend=auto
```

让 `uv` 根据 NVIDIA 驱动自动选择 PyTorch/CUDA wheel。([vLLM][3])

先：

```bash
uv --version
```

然后创建环境：

```bash
cd /mnt/data/txhan/qwen3-coder

uv venv .venv --python 3.10
```

激活：

```bash
source .venv/bin/activate
```

确认：

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

## 4. 安装 vLLM 和 Hugging Face 工具

执行：

```bash
uv pip install -U vllm --torch-backend=auto
```

然后：

```bash
uv pip install -U huggingface_hub hf_xet
```

检查：

```bash
vllm --version
hf --help
```

再检查 PyTorch 是否正常看到 8 张 H20：

```bash
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

你希望看到：

```text
CUDA available: True
GPU count: 8
0 NVIDIA H20
1 NVIDIA H20
...
```

如果这里不正常，**先不要下载模型**，先解决 CUDA/PyTorch。

---

## 5. 下载 Qwen

模型：

```text
Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8
```

官方 FP8 仓库只有约 **31.2GB**，其中三个约 10GB shard 加最后一个约 1.17GB。([Hugging Face][1])

定义路径：

```bash
export MODEL_DIR=/mnt/tydrive/txhan/models/Qwen3-Coder-30B-A3B-Instruct-FP8
```

执行：

```bash
hf download \
  Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
  --local-dir "$MODEL_DIR"
```

`hf download --local-dir` 是 Hugging Face 当前官方支持的下载方式，而且会保存下载元数据，所以掉线后重新执行同一命令可以避免从零重下。([Hugging Face][4])

### 如果中途网络断掉

直接：

```bash
hf download \
  Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
  --local-dir "$MODEL_DIR"
```

重新执行即可。

不要删目录。

---

## 6. 下载完成后检查

执行：

```bash
du -sh "$MODEL_DIR"
```

大概应该：

```text
32G
```

查看模型 shard：

```bash
ls -lh "$MODEL_DIR"/model-*.safetensors
```

应该看到四个：

```text
model-00001-of-00004.safetensors   ~10G
model-00002-of-00004.safetensors   ~10G
model-00003-of-00004.safetensors   ~10G
model-00004-of-00004.safetensors   ~1.2G
```

然后：

```bash
ls "$MODEL_DIR"
```

至少应该存在：

```text
config.json
generation_config.json
tokenizer.json
tokenizer_config.json
chat_template.jinja
model.safetensors.index.json
model-00001-of-00004.safetensors
...
```

---

## 7. 首次双卡启动（最小配置）

先不要急着加一大堆优化参数。

执行：

```bash
source /mnt/data/txhan/qwen3-coder/.venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1
export HF_HOME=/mnt/tydrive/txhan/.cache/huggingface
export TMPDIR=/mnt/data/txhan/qwen3-coder/tmp

vllm serve \
  /mnt/tydrive/txhan/models/Qwen3-Coder-30B-A3B-Instruct-FP8 \
  --served-model-name qwen3-coder \
  --tensor-parallel-size 2 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 \
  --port 8000
```

这里几个参数的意义：

```text
CUDA_VISIBLE_DEVICES=0,1
            ↓
只允许使用物理 GPU 0、1

--tensor-parallel-size 2
            ↓
一个模型横跨两张 H20

--max-model-len 65536
            ↓
先开 64K
不要第一天就开 262K

--gpu-memory-utilization 0.85
            ↓
最多使用约85%显存
留安全余量
```

---

## 8. 观察 GPU 状态

另开一个终端：

```bash
watch -n 1 nvidia-smi
```

加载期间应该看到 GPU 0/1 显存上涨。

GPU 2～7 基本保持不动：

```text
GPU0   Qwen
GPU1   Qwen

GPU2   free
GPU3   free
GPU4   free
GPU5   free
GPU6   free
GPU7   free
```

如果看到模型占用了 8 卡，说明 `CUDA_VISIBLE_DEVICES` 没生效。

---

## 9. 确认服务启动成功

vLLM 日志最后应该出现类似：

```text
Application startup complete
Uvicorn running on http://0.0.0.0:8000
```

先测试：

```bash
curl http://127.0.0.1:8000/v1/models
```

应该看到：

```json
{
  "data": [
    {
      "id": "qwen3-coder"
    }
  ]
}
```

---

## 10. 首次生成测试

执行：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-coder",
    "messages": [
      {
        "role": "user",
        "content": "Write a Python program that prints all NVIDIA GPUs and their memory usage by calling nvidia-smi."
      }
    ],
    "temperature": 0.2,
    "max_tokens": 1024
  }'
```

如果正常返回 Python 代码：

> **模型本体 + 双 H20 TP + vLLM API 已经完全跑通。**

---

## 11. 启用 Agent 工具调用

你的最终目标不是聊天，而是：

```text
Pi
 ↓
Qwen
 ↓
决定调用工具
 ↓
bash
file edit
git
pytest
 ↓
把结果返回给 Qwen
 ↓
继续下一步
```

所以在普通生成确认正常以后，停掉 vLLM：

```text
Ctrl+C
```

然后改成 Agent 版本启动：

```bash
source /mnt/data/txhan/qwen3-coder/.venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1
export HF_HOME=/mnt/tydrive/txhan/.cache/huggingface
export TMPDIR=/mnt/data/txhan/qwen3-coder/tmp

vllm serve \
  /mnt/tydrive/txhan/models/Qwen3-Coder-30B-A3B-Instruct-FP8 \
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

这里：

```text
--enable-auto-tool-choice
```

允许模型自己决定是否调用工具。

而：

```text
--tool-call-parser qwen3_xml
```

是当前 vLLM 官方对：

```text
Qwen3-Coder-30B-A3B-Instruct
```

指定的 parser。([vLLM][5])

这里**不要**加：

```bash
--reasoning-parser qwen3
```

因为 Qwen3-Coder-30B-A3B-Instruct 官方明确是 non-thinking model。([Hugging Face][2])

---

## 12. 创建启动脚本

创建：

```bash
nano /mnt/data/txhan/qwen3-coder/start_qwen.sh
```

内容：

```bash
#!/usr/bin/env bash
set -e

source /mnt/data/txhan/qwen3-coder/.venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1
export HF_HOME=/mnt/tydrive/txhan/.cache/huggingface
export TMPDIR=/mnt/data/txhan/qwen3-coder/tmp

MODEL=/mnt/tydrive/txhan/models/Qwen3-Coder-30B-A3B-Instruct-FP8

exec vllm serve "$MODEL" \
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

保存，然后：

```bash
chmod +x /mnt/data/txhan/qwen3-coder/start_qwen.sh
```

以后启动只需要：

```bash
/mnt/data/txhan/qwen3-coder/start_qwen.sh
```

---

## 13. 使用 tmux 常驻运行

创建：

```bash
tmux new -s qwen
```

然后：

```bash
/mnt/data/txhan/qwen3-coder/start_qwen.sh
```

退出但不关闭模型：

```text
Ctrl+B
然后 D
```

重新进入：

```bash
tmux attach -t qwen
```

查看：

```bash
tmux ls
```

这样 SSH 断了，Qwen 也不会跟着停。

---

## 14. 从 Windows 测试服务器

因为你现在的 Pi 是在 Windows 上用，所以需要让 Windows 能访问 Ubuntu 上的：

```text
8000
```

我更建议使用 **SSH tunnel**，而不是直接把 8000 暴露出去。

Windows PowerShell：

```powershell
ssh -N -L 8000:127.0.0.1:8000 用户名@服务器IP
```

这个窗口保持开着。

然后在另一个 PowerShell：

```powershell
curl.exe http://127.0.0.1:8000/v1/models
```

如果返回：

```text
qwen3-coder
```

就意味着：

```text
Windows
   │
   │ localhost:8000
   ↓
SSH tunnel
   ↓
Ubuntu Server
   ↓
vLLM
   ↓
2 × H20
```

已经通了。

---

## 15. 接入 Pi

Pi 当前官方支持通过：

```text
~/.pi/agent/models.json
```

添加 vLLM / OpenAI-compatible 自定义模型。([GitHub][6])

Windows 文件位置：

```text
C:\Users\TxHan\.pi\agent\models.json
```

如果没有就新建。

可以先写：

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

Pi 官方文档明确给出了这种 vLLM / OpenAI-compatible custom provider 配置方式。([GitHub][7])

然后：

```powershell
pi
```

进入 Pi 后：

```text
/model
```

应该能看到：

```text
Qwen3-Coder 30B A3B FP8 - 2xH20
```

选中它。

---

## 16. 进行真实 Agent 测试

进入你的代码项目，比如：

```powershell
cd D:\YourProject
pi
```

给它：

```text
Inspect this repository, determine how the project is structured,
run the existing tests, identify any failing tests, and explain what
needs to be fixed. Do not modify anything yet.
```

这时候你要观察它有没有：

```text
读取文件
 ↓
执行命令
 ↓
检查 Git
 ↓
跑测试
 ↓
读取 stdout/stderr
```

如果这套链路正常，才说明你真正完成了：

> **Qwen → Pi → Terminal Agent**

而不只是“模型能聊天”。

---

## 17. 稳定后将上下文从 64K 提升到 128K

不要一开始就 256K。

先：

```text
65536
```

确认稳定。

然后修改启动脚本：

```bash
--max-model-len 131072
```

Pi 同时改：

```json
"contextWindow": 131072
```

如果还是稳定，再考虑：

```bash
--max-model-len 262144
```

这个模型原生就支持 262,144 tokens，所以不需要 YaRN 才达到这个长度。([Hugging Face][2])

---

## 18. TP=1 与 TP=2 的性能说明

**2×H20 一定能很好地跑，但“双卡一定比单卡更快”并不成立。**

因为这个 FP8 checkpoint 只有约：

```text
31.2 GB
```

而你的单张 H20：

```text
96 GB
```

单卡已经能完整装下。

对于 Qwen3-Coder 这种 **3.3B active MoE**：

```text
TP=2
```

会减少每张卡计算量，但也会产生 NCCL/NVLink 通信。

所以实际可能出现：

```text
单卡：
token latency 更低

双卡：
prefill / 长上下文 / 并发更好
KV cache空间极大
```

你既然现在明确想试 **2 卡方案**，我们先用 TP=2。

等跑起来后，非常值得做一个：

```text
TP=1 vs TP=2
```

实测。

不要凭理论猜。

---

## 推荐执行顺序

```text
① nvidia-smi
        ↓
② nvidia-smi topo -m
        ↓
③ 建目录
        ↓
④ uv 创建 venv
        ↓
⑤ 安装 vLLM
        ↓
⑥ 下载官方 31.2GB FP8 Qwen
        ↓
⑦ TP=2、64K、不加 tool calling
        ↓
⑧ curl 普通对话测试
        ↓
⑨ 加 qwen3_xml tool calling
        ↓
⑩ Windows SSH tunnel
        ↓
⑪ Pi models.json
        ↓
⑫ /model 选择 qwen3-coder
        ↓
⑬ 真实 repo 测试
```

**最重要的是第 7 步先用最小参数启动。** 如果那里报错，把从执行 `vllm serve` 开始到最后报错的完整日志贴给我，我可以直接沿着你的实际 H20/CUDA/vLLM 环境往下排，不需要重新安装一遍。

[1]: https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8/tree/main "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 at main"
[2]: https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct/blob/c0ca79e77eaff38abb4b0709051148f5280fb4aa/README.md?code=true "README.md · Qwen/Qwen3-Coder-30B-A3B-Instruct at c0ca79e77eaff38abb4b0709051148f5280fb4aa"
[3]: https://docs.vllm.ai/en/latest/getting_started/installation/gpu/ "GPU - vLLM"
[4]: https://huggingface.co/docs/huggingface_hub/main/package_reference/cli "hf · Hugging Face"
[5]: https://docs.vllm.ai/en/latest/features/tool_calling/ "Tool Calling - vLLM"
[6]: https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/providers.md "pi/packages/coding-agent/docs/providers.md at main · earendil-works/pi · GitHub"
[7]: https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/models.md "pi/packages/coding-agent/docs/models.md at main · earendil-works/pi · GitHub"
