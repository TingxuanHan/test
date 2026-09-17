# DeepSeek-V4.1-Flash 单机 8×H20 部署指南

可以。下面按你现在的实际情况给一套从零到可调用 API 的完整流程。这里默认你说的是 **DeepSeek-V4.1-Flash**，机器是 **单机 8×H20**，`/mnt/tydrive` 还有约 **2.2 TB** 可用。

你现有这两项保持不动：

```bash
export HF_HOME=/mnt/tydrive/txhan/.cache/huggingface
export HF_HUB_DISABLE_XET=1
```

V4.1-Flash 官方 checkpoint 约 **511 GB / 476 GiB**，官方 vLLM recipe 给出的最低显存预算约 **614 GB**；模型支持 1,048,576 token context。([GitHub][1])

---

## 0. 先确认当前环境

执行：

```bash
echo "$HF_HOME"
echo "$HF_HUB_DISABLE_XET"

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
free -h
df -h /mnt/tydrive
df -h /
```

前两个应该是：

```text
/mnt/tydrive/txhan/.cache/huggingface
1
```

GPU 应该有 8 张 H20。

再看 GPU 拓扑：

```bash
nvidia-smi topo -m
```

---

# 1. 创建 DeepSeek 目录

我们最终使用：

```text
/mnt/tydrive/txhan/DeepSeek/
├── models/
│   └── DeepSeek-V4.1-Flash/
├── vllm_cache/
└── tmp/
```

创建：

```bash
mkdir -p /mnt/tydrive/txhan/DeepSeek/models/DeepSeek-V4.1-Flash
mkdir -p /mnt/tydrive/txhan/DeepSeek/vllm_cache
mkdir -p /mnt/tydrive/txhan/DeepSeek/tmp
```

---

# 2. 配置 DeepSeek 环境变量

**不要修改你已有的 HF_HOME。**

向 `~/.bashrc` 添加：

```bash
grep -q 'DEEPSEEK_HOME=/mnt/tydrive/txhan/DeepSeek' ~/.bashrc || cat >> ~/.bashrc <<'EOF'

# DeepSeek V4.1 Flash
export DEEPSEEK_HOME=/mnt/tydrive/txhan/DeepSeek
export MODEL_DIR=$DEEPSEEK_HOME/models/DeepSeek-V4.1-Flash
export VLLM_CACHE_DIR=$DEEPSEEK_HOME/vllm_cache
export DEEPSEEK_TMP=$DEEPSEEK_HOME/tmp

EOF
```

生效：

```bash
source ~/.bashrc
```

检查：

```bash
echo "$HF_HOME"
echo "$HF_HUB_DISABLE_XET"
echo "$DEEPSEEK_HOME"
echo "$MODEL_DIR"
echo "$VLLM_CACHE_DIR"
echo "$DEEPSEEK_TMP"
```

应该类似：

```text
/mnt/tydrive/txhan/.cache/huggingface
1
/mnt/tydrive/txhan/DeepSeek
/mnt/tydrive/txhan/DeepSeek/models/DeepSeek-V4.1-Flash
/mnt/tydrive/txhan/DeepSeek/vllm_cache
/mnt/tydrive/txhan/DeepSeek/tmp
```

注意我这里**没有全局设置 `TMPDIR`**，避免影响你服务器上 DSpark 等其他程序。

---

# 3. 安装/检查 Hugging Face CLI

先检查：

```bash
hf --version
```

如果没有：

```bash
uv tool install huggingface_hub
```

如果装完还是找不到：

```bash
export PATH="$HOME/.local/bin:$PATH"
```

再：

```bash
hf --version
```

可以永久添加：

```bash
grep -q 'HOME/.local/bin' ~/.bashrc || \
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
```

---

# 4. Hugging Face 登录可选

DeepSeek-V4.1-Flash 是公开仓库，所以不登录也可以下载。

先看：

```bash
hf auth whoami
```

如果已经登录，不用管。

没有登录又希望避免匿名下载限制：

```bash
hf auth login
```

---

# 5. 先查看要下载多少东西

建议先 dry-run：

```bash
hf download deepseek-ai/DeepSeek-V4.1-Flash \
    --local-dir "$MODEL_DIR" \
    --dry-run
```

这不会正式下载 500GB 权重。

---

# 6. 正式下载模型

因为你已经：

```bash
export HF_HUB_DISABLE_XET=1
```

继续保持即可。

建议增加超时：

```bash
export HF_HUB_DOWNLOAD_TIMEOUT=60
```

然后正式下载：

```bash
hf download deepseek-ai/DeepSeek-V4.1-Flash \
    --local-dir "$MODEL_DIR"
```

这里有个很重要的好处：

> 使用 `--local-dir` 后，大权重直接放进 `$MODEL_DIR`，不会再把完整的 511GB 模型复制一份到你现有的 `$HF_HOME`；模型目录下只会建立一个很小的 `.cache/huggingface` 元数据目录。([huggingface.co][2])

所以最后主要占空间的是：

```text
/mnt/tydrive/txhan/DeepSeek/models/DeepSeek-V4.1-Flash
```

而不是：

```text
/mnt/tydrive/txhan/.cache/huggingface
```

### SSH 容易断的话

建议在 tmux 里面下载：

```bash
tmux new -s deepseek-download
```

然后：

```bash
hf download deepseek-ai/DeepSeek-V4.1-Flash \
    --local-dir "$MODEL_DIR"
```

退出但保持任务：

```text
Ctrl+B
然后 D
```

以后重新连接：

```bash
tmux attach -t deepseek-download
```

如果下载中断，重新执行**同一条下载命令**即可。

---

# 7. 下载完成后验证

先看大小：

```bash
du -sh "$MODEL_DIR"
```

预期大约：

```text
476 GiB 左右
```

官方 recipe 给出的磁盘占用约 511GB / 476GiB。([GitHub][1])

检查：

```bash
ls -lh "$MODEL_DIR/config.json"
```

然后：

```bash
find "$MODEL_DIR" \
    -maxdepth 1 \
    -type f \
    -name "*.safetensors" | wc -l
```

再让 Hugging Face 验证文件：

```bash
hf cache verify deepseek-ai/DeepSeek-V4.1-Flash \
    --local-dir "$MODEL_DIR" \
    --fail-on-missing-files
```

---

# 8. 检查 Docker

```bash
docker --version
```

如果有输出，直接去第 9 步。

如果没有，安装 Docker：

```bash
sudo apt update
sudo apt install -y ca-certificates curl
```

```bash
sudo install -m 0755 -d /etc/apt/keyrings
```

```bash
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    -o /etc/apt/keyrings/docker.asc
```

```bash
sudo chmod a+r /etc/apt/keyrings/docker.asc
```

添加 Docker 源：

```bash
sudo tee /etc/apt/sources.list.d/docker.sources >/dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF
```

安装：

```bash
sudo apt update
sudo apt install -y \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    docker-buildx-plugin \
    docker-compose-plugin
```

当前用户加入 Docker：

```bash
sudo usermod -aG docker "$USER"
```

当前 shell 生效：

```bash
newgrp docker
```

测试：

```bash
docker ps
```

---

# 9. 检查 NVIDIA Container Toolkit

先：

```bash
nvidia-ctk --version
```

如果已有，直接配置：

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

如果没有：

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg2
```

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor \
  -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
```

```bash
curl -s -L \
  https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

然后：

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

这是 NVIDIA 当前官方 Docker 配置方式。([NVIDIA Docs][3])

---

# 10. 测试 Docker 能不能看到 8 张 H20

执行：

```bash
docker run --rm \
    --gpus all \
    nvcr.io/nvidia/cuda:12.6.2-base-ubuntu24.04 \
    nvidia-smi
```

必须看到：

```text
GPU 0 H20
GPU 1 H20
...
GPU 7 H20
```

否则先不要继续。

---

# 11. 检查 Docker 是否会吃系统盘

Docker 默认可能使用：

```text
/var/lib/docker
```

查看：

```bash
docker info --format '{{.DockerRootDir}}'
df -h /
docker system df
```

vLLM Docker 镜像本身远小于 511GB 模型，因此如果系统盘还有几十 GB 以上，一般不用动。

你的**模型、编译缓存、临时文件都已经在 `/mnt/tydrive`**。

---

# 12. 拉取 vLLM nightly

这一点现在很重要。

DeepSeek-V4.1-Flash 官方 vLLM recipe 当前要求 **vLLM ≥0.30.0**，但 0.30.0 尚未正式发布，官方明确要求 NVIDIA 路线使用 nightly；普通旧 stable image 不建议用于该架构。([GitHub][1])

执行：

```bash
docker pull vllm/vllm-openai:nightly
```

检查：

```bash
docker images | grep vllm
```

再确认这个 image 能看到 GPU：

```bash
docker run --rm \
    --gpus all \
    --entrypoint nvidia-smi \
    vllm/vllm-openai:nightly
```

---

# 13. 第一次启动：先求稳定

第一次不要：

```text
1M context
DSpark
高并发
图像输入
```

先用：

```text
8×H20
TP8
EP
纯文本
64K context
32 并发上限
不开 DSpark
eager mode
```

官方 V4.1-Flash recipe支持 `deepseek_v41` tokenizer、reasoning parser、tool parser 以及 `--language-model-only`。([GitHub][1])

执行：

```bash
docker run -d \
    --name deepseek-v41-flash \
    --gpus all \
    --ipc=host \
    --network=host \
    -v "$MODEL_DIR:/model:ro" \
    -v "$HF_HOME:/root/.cache/huggingface" \
    -v "$VLLM_CACHE_DIR:/root/.cache/vllm" \
    -v "$DEEPSEEK_TMP:/deepseek_tmp" \
    -e TMPDIR=/deepseek_tmp \
    -e HF_HUB_DISABLE_XET=1 \
    -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    vllm/vllm-openai:nightly \
    --model /model \
    --served-model-name deepseek-v41-flash \
    --host 127.0.0.1 \
    --port 8000 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --language-model-only \
    --tokenizer-mode deepseek_v41 \
    --tool-call-parser deepseek_v41 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v41 \
    --max-model-len 65536 \
    --max-num-seqs 32 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.90 \
    --enforce-eager
```

我这里用：

```bash
--host 127.0.0.1
```

意味着暂时只有本机能访问 API，更安全。

如果之后你要让另一台机器/Pi 调用，再改成：

```bash
--host 0.0.0.0
```

---

# 14. 查看模型加载

看日志：

```bash
docker logs -f deepseek-v41-flash
```

另开一个终端：

```bash
watch -n 1 nvidia-smi
```

再看 container：

```bash
docker ps
```

如果 container 退出：

```bash
docker ps -a
docker logs --tail 300 deepseek-v41-flash
```

---

# 15. 检查 API

模型加载完成后：

```bash
curl http://127.0.0.1:8000/health
```

再：

```bash
curl http://127.0.0.1:8000/v1/models
```

应该看到：

```text
deepseek-v41-flash
```

---

# 16. 第一次生成测试

V4.1-Flash 默认如果什么都不设置，会开启 thinking，默认 effort 50；小 `max_tokens` 时甚至可能全部被思考过程吃掉，因此 smoke test 最好明确关闭 thinking。([GitHub][1])

执行：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v41-flash",
    "messages": [
      {
        "role": "user",
        "content": "What is 17 * 19? Return only the integer."
      }
    ],
    "temperature": 1.0,
    "max_tokens": 128,
    "chat_template_kwargs": {
      "thinking": false
    }
  }'
```

应该返回：

```text
323
```

---

# 17. 再测试 reasoning

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v41-flash",
    "messages": [
      {
        "role": "user",
        "content": "Explain speculative decoding and why it can accelerate LLM inference."
      }
    ],
    "temperature": 1.0,
    "top_p": 0.95,
    "max_tokens": 2048,
    "chat_template_kwargs": {
      "thinking": true,
      "reasoning_effort": 25
    }
  }'
```

这个模型本地 vLLM 支持 reasoning effort：

```text
low   = 25
high  = 50
xhigh = 75
max   = 100
```

或者直接传整数 1–100。([GitHub][1])

---

# 18. Smoke test 稳定后，去掉 eager

第一次稳定后：

```bash
docker stop deepseek-v41-flash
docker rm deepseek-v41-flash
```

重新执行上面的启动命令，但删掉：

```bash
--enforce-eager
```

这样让 vLLM 使用正常的高性能执行路径。

---

# 19. 再逐渐扩大 context

我建议依次：

```text
65536
→ 131072
→ 262144
→ 524288
→ 1048576
```

例如把：

```bash
--max-model-len 65536
```

改成：

```bash
--max-model-len 131072
```

先不要直接跳 1M。

V4.1-Flash 官方支持最大 **1,048,576 tokens**。([GitHub][1])

---

# 20. 最后才开启 DSpark

模型完全稳定后，增加：

```bash
--speculative-config \
'{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":true}'
```

官方 V4.1 recipe 的 DSpark 就是一次 draft **5 tokens**。([GitHub][1])

也就是说后期可以变成：

```bash
...
--max-model-len 131072 \
--max-num-seqs 32 \
--max-num-batched-tokens 4096 \
--gpu-memory-utilization 0.90 \
--speculative-config \
'{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":true}'
```

---

# 21. H20 上暂时不要把并发开太高

目前 vLLM 官方 recipe 的正式 verified hardware 列表还没有 H20，但已有 **8×H20、TP8、EP8、DSpark** 的实际运行报告。该报告发现高并发、`max_num_seqs > 256` 时可能在 `dsv4_topk` 出现 CUDA illegal memory access，而限制到 256 后不再复现。([GitHub][4])

所以你的测试顺序建议：

```text
32
→ 64
→ 128
→ 256
```

现阶段：

```bash
--max-num-seqs 32
```

是非常稳妥的起点。

---

## 你最终的磁盘布局

完成后大概是：

```text
/mnt/tydrive/txhan/
│
├── .cache/
│   └── huggingface/
│       └── 你原来已有的 HF 缓存
│
└── DeepSeek/
    ├── models/
    │   └── DeepSeek-V4.1-Flash/
    │       ├── config.json
    │       ├── *.safetensors
    │       ├── encoding/
    │       └── ...
    │
    ├── vllm_cache/
    │   └── 编译/JIT缓存
    │
    └── tmp/
```

你目前 **2.2 TB 可用**，模型本身约 **511 GB**，所以单独部署这一套空间非常充裕。([GitHub][1])

**你现在实际应该从第 0～7 步先做，把 511GB 模型完整下载并 `hf cache verify` 通过；然后再进行 Docker/vLLM 部署。**

[1]: https://github.com/vllm-project/recipes/blob/main/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml "recipes/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml at main · vllm-project/recipes · GitHub"
[2]: https://huggingface.co/docs/huggingface_hub/guides/download?utm_source=chatgpt.com "Download files from the Hub · Hugging Face"
[3]: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html?utm_source=chatgpt.com "Installing the NVIDIA Container Toolkit — NVIDIA Container Toolkit"
[4]: https://github.com/vllm-project/vllm/issues/56389?utm_source=chatgpt.com "[Bug]: DeepSeek-V4.1-Flash dsv4_topk Triton illegal memory access under high concurrency on H20; mitigated by max_num_seqs=256 · Issue #56389 · vllm-project/vllm · GitHub"

