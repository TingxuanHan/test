# DeepSeek-V4.1-Flash 单机 8×H20 部署指南

可以。下面按你现在的实际情况给一套从零到可调用 API 的完整流程。这里默认你说的是 **DeepSeek-V4.1-Flash**，机器是 **单机 8×H20**，`/mnt/tydrive` 还有约 **2.2 TB** 可用。

本文把所有主要大文件路径统一放到<code>/mnt/tydrive</code>：模型、Hugging Face缓存、uv/pip缓存、vLLM编译缓存、临时文件，以及Docker和containerd的镜像层、容器层、卷与构建缓存。系统盘只保留软件包、少量配置和运行时状态。

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
findmnt -T /mnt/tydrive
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

/mnt/tydrive/txhan/.cache/
├── huggingface/
├── pip/
└── uv/

/mnt/tydrive/txhan/.local/share/uv/tools/

/mnt/tydrive/docker-data/       # Docker daemon数据，root拥有
/mnt/tydrive/containerd-data/   # Docker 29+ containerd镜像存储，root拥有
```

创建：

```bash
mkdir -p /mnt/tydrive/txhan/DeepSeek/models/DeepSeek-V4.1-Flash
mkdir -p /mnt/tydrive/txhan/DeepSeek/vllm_cache
mkdir -p /mnt/tydrive/txhan/DeepSeek/tmp
mkdir -p /mnt/tydrive/txhan/.cache/huggingface
mkdir -p /mnt/tydrive/txhan/.cache/pip
mkdir -p /mnt/tydrive/txhan/.cache/uv
mkdir -p /mnt/tydrive/txhan/.local/share/uv/tools
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
export PIP_CACHE_DIR=/mnt/tydrive/txhan/.cache/pip
export UV_CACHE_DIR=/mnt/tydrive/txhan/.cache/uv
export UV_TOOL_DIR=/mnt/tydrive/txhan/.local/share/uv/tools

EOF

grep -qxF 'export PIP_CACHE_DIR=/mnt/tydrive/txhan/.cache/pip' ~/.bashrc \
  || echo 'export PIP_CACHE_DIR=/mnt/tydrive/txhan/.cache/pip' >> ~/.bashrc

grep -qxF 'export UV_CACHE_DIR=/mnt/tydrive/txhan/.cache/uv' ~/.bashrc \
  || echo 'export UV_CACHE_DIR=/mnt/tydrive/txhan/.cache/uv' >> ~/.bashrc

grep -qxF 'export UV_TOOL_DIR=/mnt/tydrive/txhan/.local/share/uv/tools' ~/.bashrc \
  || echo 'export UV_TOOL_DIR=/mnt/tydrive/txhan/.local/share/uv/tools' >> ~/.bashrc
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
echo "$PIP_CACHE_DIR"
echo "$UV_CACHE_DIR"
echo "$UV_TOOL_DIR"
```

应该类似：

```text
/mnt/tydrive/txhan/.cache/huggingface
1
/mnt/tydrive/txhan/DeepSeek
/mnt/tydrive/txhan/DeepSeek/models/DeepSeek-V4.1-Flash
/mnt/tydrive/txhan/DeepSeek/vllm_cache
/mnt/tydrive/txhan/DeepSeek/tmp
/mnt/tydrive/txhan/.cache/pip
/mnt/tydrive/txhan/.cache/uv
/mnt/tydrive/txhan/.local/share/uv/tools
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

确认uv缓存和工具环境都位于数据盘：

~~~bash
uv cache dir
uv tool dir
~~~

应分别指向：

~~~text
/mnt/tydrive/txhan/.cache/uv
/mnt/tydrive/txhan/.local/share/uv/tools
~~~

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

Docker和NVIDIA Container Toolkit安装完成后，可以清理APT下载缓存；软件本身仍正常保留：

~~~bash
sudo apt-get clean
sudo du -sh /var/cache/apt 2>/dev/null || true
~~~

---

# 10. 将 Docker 和 containerd 大数据迁移到 /mnt/tydrive

这一步必须在第一次拉取CUDA或vLLM镜像前完成。Docker官方支持通过<code>data-root</code>迁移Docker daemon数据；但Docker Engine 29.0及以后新安装默认可能启用containerd image store，此时镜像内容和容器快照位于<code>/var/lib/containerd</code>，只修改<code>data-root</code>还不够。([Docker Docs][5])

本文统一使用：

~~~text
/mnt/tydrive/docker-data
/mnt/tydrive/containerd-data
~~~

这两个目录是系统级容器存储，必须由root管理，不要放进普通用户可随意删除的项目目录。

### 10.1 确认数据盘和现有容器状态

先确认<code>/mnt/tydrive</code>确实是独立且持久挂载的数据盘，不是系统盘上的普通空目录：

~~~bash
findmnt -T /mnt/tydrive
df -hT / /mnt/tydrive
mountpoint /mnt/tydrive
grep -F '/mnt/tydrive' /etc/fstab || true
~~~

<code>mountpoint</code>必须成功，<code>findmnt</code>应显示真实块设备或预期文件系统。如果该挂载没有写入<code>/etc/fstab</code>或其他可靠的开机挂载配置，先修复持久挂载，再迁移Docker。

记录当前状态：

~~~bash
mkdir -p /mnt/tydrive/txhan/DeepSeek/migration-backup

docker info --format 'DockerRootDir={{.DockerRootDir}}'
docker info --format 'Driver={{.Driver}} DriverStatus={{json .DriverStatus}}'
docker ps -a --no-trunc \
  | tee /mnt/tydrive/txhan/DeepSeek/migration-backup/docker-ps-before.txt
docker images --digests \
  | tee /mnt/tydrive/txhan/DeepSeek/migration-backup/docker-images-before.txt

sudo du -sh /var/lib/docker /var/lib/containerd 2>/dev/null || true
sudo ctr namespaces list 2>/dev/null || true
~~~

如果<code>ctr namespaces list</code>显示Kubernetes或其他非Docker工作负载，先确认这些服务的维护窗口和迁移方案，不要直接执行后续containerd迁移。

### 10.2 停止服务并复制现有数据

安装rsync并清理APT缓存：

~~~bash
command -v rsync >/dev/null || sudo apt-get install -y rsync
sudo apt-get clean
~~~

停止Docker和containerd：

~~~bash
sudo systemctl stop docker.service docker.socket
sudo systemctl stop containerd.service

sudo systemctl is-active docker.service containerd.service || true
~~~

创建目标目录：

~~~bash
sudo install -d -m 0711 -o root -g root /mnt/tydrive/docker-data
sudo install -d -m 0711 -o root -g root /mnt/tydrive/containerd-data
~~~

复制已有数据。源目录不存在或为空时，对应命令会自动跳过：

~~~bash
if [ -d /var/lib/docker ]; then
  sudo rsync -aHAX --numeric-ids --info=progress2 \
    /var/lib/docker/ /mnt/tydrive/docker-data/
fi

if [ -d /var/lib/containerd ]; then
  sudo rsync -aHAX --numeric-ids --info=progress2 \
    /var/lib/containerd/ /mnt/tydrive/containerd-data/
fi
~~~

此时不要删除<code>/var/lib/docker</code>或<code>/var/lib/containerd</code>，它们仍是回滚副本。

### 10.3 备份并合并Docker配置

NVIDIA Container Toolkit可能已经在<code>/etc/docker/daemon.json</code>中写入runtime配置，因此不能用一个新的JSON文件直接覆盖。先备份：

~~~bash
STAMP="$(date +%Y%m%d-%H%M%S)"
sudo install -d -m 0755 /mnt/tydrive/txhan/DeepSeek/migration-backup

if [ -f /etc/docker/daemon.json ]; then
  sudo cp -a /etc/docker/daemon.json \
    "/mnt/tydrive/txhan/DeepSeek/migration-backup/daemon.json.$STAMP"
fi

if [ -f /etc/containerd/config.toml ]; then
  sudo cp -a /etc/containerd/config.toml \
    "/mnt/tydrive/txhan/DeepSeek/migration-backup/containerd-config.toml.$STAMP"
fi
~~~

用Python合并Docker配置，保留现有NVIDIA runtime，同时增加数据目录和容器日志轮转：

~~~bash
sudo python3 - <<'PY'
import json
import os

path = "/etc/docker/daemon.json"
data = {}

if os.path.exists(path) and os.path.getsize(path) > 0:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

data["data-root"] = "/mnt/tydrive/docker-data"
data["log-driver"] = "local"
data["log-opts"] = {
    "max-size": "100m",
    "max-file": "5"
}

os.makedirs(os.path.dirname(path), exist_ok=True)
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
    f.write("\n")
os.chmod(tmp, 0o644)
os.replace(tmp, path)
PY
~~~

Docker的<code>local</code>日志驱动自带轮转和压缩；这里再把单个容器日志限制为最多5个、每个100MB。该默认值只对之后新建的容器生效。([Docker Docs][6])

### 10.4 配置containerd数据目录

如果没有containerd配置文件，先从当前版本生成默认配置：

~~~bash
sudo install -d -m 0755 /etc/containerd

if [ ! -s /etc/containerd/config.toml ]; then
  containerd config default \
    | sudo tee /etc/containerd/config.toml >/dev/null
fi
~~~

只修改顶层<code>root</code>，运行时<code>state</code>仍保留在<code>/run/containerd</code>：

~~~bash
sudo python3 - <<'PY'
import os
import re

path = "/etc/containerd/config.toml"
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

root_line = 'root = "/mnt/tydrive/containerd-data"'
pattern = re.compile(r"(?m)^root\s*=\s*['\"][^'\"]*['\"]\s*$")

if pattern.search(text):
    text = pattern.sub(root_line, text, count=1)
else:
    version_pattern = re.compile(r"(?m)^version\s*=.*$")
    match = version_pattern.search(text)
    if match:
        text = text[:match.end()] + "\n" + root_line + text[match.end():]
    else:
        text = root_line + "\n" + text

tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    f.write(text)
os.chmod(tmp, 0o644)
os.replace(tmp, path)
PY
~~~

验证两个配置文件：

~~~bash
sudo dockerd --validate --config-file=/etc/docker/daemon.json
sudo containerd --config /etc/containerd/config.toml config dump >/dev/null

sudo grep -nE '"data-root"|"log-driver"|"max-size"|"max-file"' \
  /etc/docker/daemon.json
sudo grep -n '^root[[:space:]]*=' /etc/containerd/config.toml
~~~

### 10.5 防止数据盘未挂载时误写系统盘

为Docker和containerd增加systemd挂载依赖及启动前检查：

~~~bash
sudo install -d -m 0755 /etc/systemd/system/docker.service.d
sudo install -d -m 0755 /etc/systemd/system/containerd.service.d

sudo tee /etc/systemd/system/docker.service.d/tydrive.conf >/dev/null <<'EOF'
[Unit]
RequiresMountsFor=/mnt/tydrive

[Service]
ExecStartPre=/usr/bin/mountpoint -q /mnt/tydrive
EOF

sudo tee /etc/systemd/system/containerd.service.d/tydrive.conf >/dev/null <<'EOF'
[Unit]
RequiresMountsFor=/mnt/tydrive

[Service]
ExecStartPre=/usr/bin/mountpoint -q /mnt/tydrive
EOF

sudo systemctl daemon-reload
sudo systemctl start containerd.service
sudo systemctl start docker.service
~~~

检查服务与路径：

~~~bash
sudo systemctl --no-pager --full status containerd.service docker.service

docker info --format 'DockerRootDir={{.DockerRootDir}}'
docker info --format 'LoggingDriver={{.LoggingDriver}}'
sudo grep -n '^root[[:space:]]*=' /etc/containerd/config.toml

df -hT / /mnt/tydrive
sudo du -sh /mnt/tydrive/docker-data /mnt/tydrive/containerd-data
~~~

目标结果：

~~~text
DockerRootDir=/mnt/tydrive/docker-data
LoggingDriver=local
root = "/mnt/tydrive/containerd-data"
~~~

### 10.6 验证后再回收系统盘旧数据

先完成第11～16步、重启一次服务器，并再次确认Docker、GPU和DeepSeek容器都正常。验证期内保留原目录可快速回滚。

稳定后如果要释放系统盘，可先把旧目录移到数据盘作为临时备份，而不是直接删除：

~~~bash
sudo systemctl stop docker.service docker.socket
sudo systemctl stop containerd.service

STAMP="$(date +%Y%m%d-%H%M%S)"

if [ -d /var/lib/docker ]; then
  sudo mv /var/lib/docker "/mnt/tydrive/docker-before-migration.$STAMP"
fi

if [ -d /var/lib/containerd ]; then
  sudo mv /var/lib/containerd "/mnt/tydrive/containerd-before-migration.$STAMP"
fi

sudo systemctl start containerd.service
sudo systemctl start docker.service

df -h /
~~~

跨文件系统的<code>mv</code>会复制后再删除源目录，数据较大时需要等待。确认新目录长期稳定且不再需要回滚后，再人工处理这些数据盘备份。不要在首次迁移时直接<code>rm -rf /var/lib/docker</code>或<code>rm -rf /var/lib/containerd</code>。

如果服务启动失败，保持Docker停止，恢复第10.3节备份的<code>daemon.json</code>和<code>config.toml</code>，执行<code>sudo systemctl daemon-reload</code>后重新启动。原系统盘目录尚未清理时，回滚不会丢失原镜像和容器。

---

# 11. 测试 Docker 能否看到 8 张 H20并复核磁盘

现在执行的第一次镜像拉取应直接写入<code>/mnt/tydrive</code>：

~~~bash
docker run --rm \
    --gpus all \
    nvcr.io/nvidia/cuda:12.6.2-base-ubuntu24.04 \
    nvidia-smi
~~~

必须看到：

~~~text
GPU 0 H20
GPU 1 H20
...
GPU 7 H20
~~~

否则先不要继续。

再次复核容器存储和系统盘：

~~~bash
docker info --format 'DockerRootDir={{.DockerRootDir}}'
docker info --format 'Driver={{.Driver}} DriverStatus={{json .DriverStatus}}'
docker info --format 'LoggingDriver={{.LoggingDriver}}'
docker system df

sudo du -sh /mnt/tydrive/docker-data /mnt/tydrive/containerd-data
sudo du -sh /var/lib/docker /var/lib/containerd 2>/dev/null || true
df -hT / /mnt/tydrive
~~~

必须确认Docker root位于<code>/mnt/tydrive/docker-data</code>。如果<code>DriverStatus</code>显示<code>io.containerd.snapshotter.v1</code>，镜像和快照还必须落在<code>/mnt/tydrive/containerd-data</code>。此后Docker镜像、容器可写层、卷、BuildKit缓存和容器日志都不再以系统盘为主要存储位置。

---
# 12. 拉取 vLLM nightly

这一点现在很重要。

拉取前必须再次确认Docker和containerd都已切换到数据盘：

~~~bash
docker info --format 'DockerRootDir={{.DockerRootDir}}'
sudo grep -n '^root[[:space:]]*=' /etc/containerd/config.toml
~~~

如果仍显示<code>/var/lib/docker</code>或<code>/var/lib/containerd</code>，先返回第10步修复，不要继续拉取大镜像。

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

完成后大致是：

~~~text
/mnt/tydrive/
│
├── docker-data/                         # Docker daemon数据
│   ├── containers/
│   ├── volumes/
│   ├── buildkit/
│   └── ...
│
├── containerd-data/                     # Docker 29+镜像内容和快照
│   ├── io.containerd.content.v1.content/
│   ├── io.containerd.snapshotter.v1.overlayfs/
│   └── ...
│
├── docker-before-migration.*            # 可选临时回滚副本
├── containerd-before-migration.*        # 可选临时回滚副本
│
└── txhan/
    ├── .cache/
    │   ├── huggingface/
    │   ├── pip/
    │   └── uv/
    │
    ├── .local/share/uv/tools/             # uv tool环境
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
        ├── migration-backup/
        │   ├── daemon.json.*
        │   ├── containerd-config.toml.*
        │   └── 迁移前清单
        │
        └── tmp/
~~~

系统盘仍会保留Docker、containerd和NVIDIA Container Toolkit的软件包，以及<code>/etc/docker</code>、<code>/etc/containerd</code>和systemd配置；这些体积很小且属于系统配置，不应迁移。主要增长项已经转到<code>/mnt/tydrive</code>。

## 最终磁盘验收与系统盘清理

执行：

~~~bash
docker info --format 'DockerRootDir={{.DockerRootDir}}'
docker info --format 'Driver={{.Driver}} DriverStatus={{json .DriverStatus}}'
docker info --format 'LoggingDriver={{.LoggingDriver}}'
sudo grep -n '^root[[:space:]]*=' /etc/containerd/config.toml

sudo du -sh \
  /mnt/tydrive/docker-data \
  /mnt/tydrive/containerd-data \
  /mnt/tydrive/txhan/DeepSeek \
  /mnt/tydrive/txhan/.cache

df -hT / /mnt/tydrive
docker system df
~~~

还可以只读检查系统盘剩余的大目录：

~~~bash
sudo du -xhd1 /var 2>/dev/null | sort -h
sudo du -xhd1 /root 2>/dev/null | sort -h
du -xhd1 "$HOME" 2>/dev/null | sort -h
journalctl --disk-usage
~~~

APT缓存可以安全清理：

~~~bash
sudo apt-get clean
~~~

如果systemd journal明显过大，并且已经保留完需要的故障证据，可选择限制历史日志总量：

~~~bash
sudo journalctl --vacuum-size=1G
~~~

该命令会删除较旧的journal日志，不要在仍需调查故障时执行。不要为了腾空间运行<code>docker system prune -a --volumes</code>；它可能删除仍需使用的镜像、缓存和卷。

最终应同时满足：

- 模型位于<code>/mnt/tydrive/txhan/DeepSeek/models</code>；
- Hugging Face、pip、uv、uv tool和vLLM数据位于<code>/mnt/tydrive</code>；
- Docker root为<code>/mnt/tydrive/docker-data</code>；
- containerd root为<code>/mnt/tydrive/containerd-data</code>；
- Docker日志驱动为带轮转的<code>local</code>；
- 数据盘未挂载时Docker和containerd拒绝启动；
- 第一次CUDA和vLLM镜像拉取发生在迁移完成之后；
- 系统盘旧Docker/containerd目录只在验证期临时保留，之后按第10.6节移到数据盘。

你目前<code>/mnt/tydrive</code>约有2.2TB可用，模型本身约511GB；迁移前仍需结合现有Docker镜像和containerd数据量确认剩余空间。([GitHub][1])

实际执行顺序应为：先完成第0～7步并验证模型，再完成第8～11步的Docker安装、存储迁移和GPU检查；只有Docker与containerd路径都验证为<code>/mnt/tydrive</code>后，才从第12步开始拉取vLLM镜像。

[1]: https://github.com/vllm-project/recipes/blob/main/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml "DeepSeek-V4.1-Flash vLLM recipe"
[2]: https://huggingface.co/docs/huggingface_hub/guides/download "Hugging Face Hub download guide"
[3]: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html "NVIDIA Container Toolkit install guide"
[4]: https://github.com/vllm-project/vllm/issues/56389 "DeepSeek-V4.1-Flash H20 high-concurrency issue"
[5]: https://docs.docker.com/engine/daemon/#daemon-data-directory "Docker daemon data directory"
[6]: https://docs.docker.com/engine/logging/drivers/local/ "Docker local logging driver"
[7]: https://docs.astral.sh/uv/reference/storage/ "uv storage locations"
