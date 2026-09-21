# `/root`目录安全清理与缓存迁移指南

本指南按**空间检查→安全清理→Conda清理→定位剩余大文件→迁移缓存→最终验证**的顺序执行。

> [!WARNING]
> 本文包含`rm -rf`、Conda环境删除、`find -delete`和`apt autoremove`等破坏性命令。每次执行前必须重新确认目标路径、当前进程和文件用途；不确定时先停止，或将目标移动到备份目录。不要把历史容量数字当作当前状态。

## 1. 先记录当前空间

```bash
df -h /
```

看 `/root` 总体：

```bash
sudo du -xhd1 /root 2>/dev/null | sort -h
```

再看两层：

```bash
sudo du -xhd2 /root 2>/dev/null | sort -h | tail -50
```

---

## 2. 复核并清理已确认的uv和pip缓存

此前检查结果为：

```text
/root/.cache/uv   41G
/root/.cache/pip  46G
```

执行前重新确认当前大小：

```bash
sudo du -sh /root/.cache/uv /root/.cache/pip 2>/dev/null
```

> [!CAUTION]
> 确认当前没有`uv`或`pip`安装任务运行，且目标仍然是缓存目录后，再执行以下命令。

```bash
sudo rm -rf /root/.cache/uv
sudo rm -rf /root/.cache/pip
```

确认：

```bash
df -h /
sudo du -sh /root/.cache 2>/dev/null
```

理论上这里应该立即释放接近 **87G**。

---

## 3. 暂时不要删 Hugging Face 的 47G

因为你还在下载 DeepSeek V4 Flash。

先看它里面是什么：

```bash
sudo du -xhd3 /root/.cache/huggingface 2>/dev/null | sort -h | tail -50
```

找 DeepSeek：

```bash
sudo find /root/.cache/huggingface \
  -maxdepth 5 \
  -iname '*deepseek*' \
  -print 2>/dev/null
```

查未完成下载：

```bash
sudo find /root/.cache/huggingface \
  -type f \
  \( -name '*.incomplete' -o -name '*.lock' \) \
  -ls 2>/dev/null
```

如果里面有你正在下载的 DeepSeek：

> **先保留 `/root/.cache/huggingface`。**

不要执行删除。

如果以后 DeepSeek 已经完整下载到：

```text
/mnt/tydrive/txhan/...
```

并确认root下只是旧缓存，再考虑删除。

> [!WARNING]
> 只要DeepSeek下载仍可能依赖`/root/.cache/huggingface`，就不要执行下面的命令。必须先确认模型已完整下载到数据盘，并完成完整性验证。

```bash
sudo rm -rf /root/.cache/huggingface
```

现在先跳过这一步。

---

## 4. 清理 `/root/miniconda3`

你这里有约：

```text
205G
```

先看组成：

```bash
sudo du -xhd1 /root/miniconda3 2>/dev/null | sort -h
```

通常最大的会是：

```text
/root/miniconda3/pkgs
/root/miniconda3/envs
```

先用 Conda 自己清缓存：

```bash
sudo /root/miniconda3/bin/conda clean -a -y
```

清完：

```bash
sudo du -sh /root/miniconda3
df -h /
```

这一步**不会删除你的 Conda 环境**，主要清 package cache、tarball、index cache 等。

---

## 5. 检查有哪些 Conda 环境

```bash
sudo /root/miniconda3/bin/conda env list
```

再看各环境实际大小：

```bash
sudo du -xhd1 /root/miniconda3/envs 2>/dev/null | sort -h
```

例如可能出现：

```text
18G   /root/miniconda3/envs/a
40G   /root/miniconda3/envs/old-vllm
65G   /root/miniconda3/envs/old-deepseek
```

如果确认某个旧环境不再使用：

> [!CAUTION]
> 先核对环境的完整绝对路径，并确认没有进程正在使用该环境。只删除已明确废弃的单个环境。

```bash
sudo /root/miniconda3/bin/conda env remove \
  -p /root/miniconda3/envs/环境名称 \
  -y
```

例如：

```bash
sudo /root/miniconda3/bin/conda env remove \
  -p /root/miniconda3/envs/old-vllm \
  -y
```

**不要直接删除整个 `/root/miniconda3`。**

---

## 6. 查 `/root` 剩下那约 300G

重新：

```bash
sudo du -xhd1 /root 2>/dev/null | sort -h
```

然后：

```bash
sudo du -xhd2 /root 2>/dev/null | sort -h | tail -60
```

再直接找超过 5G 的单文件：

```bash
sudo find /root -xdev -type f -size +5G \
  -printf '%s %p\n' 2>/dev/null \
  | sort -n \
  | numfmt --field=1 --to=iec
```

这一步很重要。

因为你之前：

```text
/root 总计 642G
.cache      134G
miniconda3  205G
```

还有大约：

```text
303G
```

尚未解释。

---

## 7. 一次检查常见大目录

运行：

```bash
sudo du -sh \
  /root/.local \
  /root/.nv \
  /root/.triton \
  /root/.conda \
  /root/.vscode-server \
  /root/.cargo \
  /root/.npm \
  /root/.cache \
  /root/snap \
  2>/dev/null | sort -h
```

重点关注：

```text
/root/.triton
/root/.cache/vllm
/root/.cache/torch_extensions
/root/.nv
/root/.local
```

---

## 8. 清理旧的 AI 编译缓存

先看：

```bash
sudo du -sh /root/.triton 2>/dev/null
sudo du -sh /root/.cache/vllm 2>/dev/null
sudo du -sh /root/.cache/torch_extensions 2>/dev/null
```

如果确认它们只是旧的Qwen/vLLM编译缓存，而且当前没有vLLM或相关编译任务运行：

> [!CAUTION]
> 这些是root用户级共享缓存，可能被其他模型或项目复用。执行前逐项核对目录内容；不确定时先移动到备份位置。

```bash
sudo rm -rf /root/.triton
sudo rm -rf /root/.cache/vllm
sudo rm -rf /root/.cache/torch_extensions
```

这些删除后以后需要时会重新生成。

不会删除你的模型。

---

## 9. 检查 VS Code Server

你长期用 VS Code Remote 的话，这个也可能比较大：

```bash
sudo du -xhd2 /root/.vscode-server 2>/dev/null | sort -h | tail -30
```

特别看：

```text
/root/.vscode-server/extensions
/root/.vscode-server/bin
/root/.vscode-server/cli
```

**暂时不要直接删整个 `.vscode-server`**，否则远程 VS Code 可能重新安装服务器组件。

如果里面存在很多旧版本 server，可以后面针对旧版本清。

---

## 10. 检查 npm 缓存

```bash
sudo du -sh /root/.npm 2>/dev/null
```

如果很大：

```bash
sudo npm cache clean --force
```

如果只是缓存而且 npm 命令不可用，也可以：

```bash
sudo rm -rf /root/.npm/_cacache
```

不要把 `/root/.npm` 整个删掉，先只清 cache。

---

## 11. 检查 `.local`

```bash
sudo du -xhd2 /root/.local 2>/dev/null | sort -h | tail -40
```

注意：

```text
/root/.local/share/uv
/root/.local/bin
```

不要看到大就直接删。

特别是：

```text
/root/.local/share/uv/python
```

可能是 uv 下载管理的 Python。

通常不会几百 GB。

---

## 12. 清理系统临时目录

先看：

```bash
sudo du -sh /tmp /var/tmp 2>/dev/null
```

列出大的：

```bash
sudo du -xhd1 /tmp 2>/dev/null | sort -h | tail -30
sudo du -xhd1 /var/tmp 2>/dev/null | sort -h | tail -30
```

先预览满足时间条件的目标：

```bash
sudo find /tmp -mindepth 1 -mtime +3 -print
sudo find /var/tmp -mindepth 1 -mtime +7 -print
```

> [!CAUTION]
> 确认输出中没有正在使用的socket、锁文件、服务临时目录或当前安装任务文件后，才执行删除。

```bash
sudo find /tmp -mindepth 1 -mtime +3 -delete
sudo find /var/tmp -mindepth 1 -mtime +7 -delete
```

这个比：

```bash
rm -rf /tmp/*
```

安全得多。

---

## 13. 清系统包缓存

Ubuntu 可以：

```bash
sudo apt clean
```

先预览将被移除的软件包：

```bash
sudo apt autoremove --dry-run
```

确认列表无误后再执行：

```bash
sudo apt autoremove -y
```

然后：

```bash
df -h /
```

这一块通常不会释放几十 GB，但可以顺手整理。

---

## 14. 把以后所有 AI/Python 缓存迁出 `/root`

你现在的核心问题就是很多工具默认写：

```text
/root/.cache
```

创建新的缓存目录：

```bash
mkdir -p /mnt/data/txhan/.cache/uv
mkdir -p /mnt/data/txhan/.cache/pip
mkdir -p /mnt/data/txhan/.cache/vllm
mkdir -p /mnt/data/txhan/.cache/torch_extensions
mkdir -p /mnt/data/txhan/.cache/triton

mkdir -p /mnt/tydrive/txhan/.cache/huggingface
```

---

## 15. 修改 `/root/.bashrc`

先备份：

```bash
sudo cp -a /root/.bashrc /root/.bashrc.before-cache-move
```

再编辑：

```bash
sudo nano /root/.bashrc
```

在最后加入：

```bash
# ==============================
# AI / Python cache directories
# ==============================

# uv
export UV_CACHE_DIR=/mnt/data/txhan/.cache/uv

# pip
export PIP_CACHE_DIR=/mnt/data/txhan/.cache/pip

# Hugging Face
export HF_HOME=/mnt/tydrive/txhan/.cache/huggingface

# vLLM
export VLLM_CACHE_ROOT=/mnt/data/txhan/.cache/vllm

# PyTorch CUDA extensions
export TORCH_EXTENSIONS_DIR=/mnt/data/txhan/.cache/torch_extensions

# Triton
export TRITON_CACHE_DIR=/mnt/data/txhan/.cache/triton
```

保存后：

```bash
source /root/.bashrc
```

---

## 16. 验证变量

```bash
echo $UV_CACHE_DIR
echo $PIP_CACHE_DIR
echo $HF_HOME
echo $VLLM_CACHE_ROOT
echo $TORCH_EXTENSIONS_DIR
echo $TRITON_CACHE_DIR
```

应该分别指向：

```text
/mnt/data/txhan/.cache/uv
/mnt/data/txhan/.cache/pip
/mnt/tydrive/txhan/.cache/huggingface
/mnt/data/txhan/.cache/vllm
/mnt/data/txhan/.cache/torch_extensions
/mnt/data/txhan/.cache/triton
```

---

## 17. 特别检查 uv

```bash
uv cache dir
```

应该是：

```text
/mnt/data/txhan/.cache/uv
```

如果仍显示：

```text
/root/.cache/uv
```

说明环境变量没有生效。

---

## 18. 特别检查 Hugging Face

```bash
python - <<'PY'
import os
print(os.environ.get("HF_HOME"))
PY
```

应该：

```text
/mnt/tydrive/txhan/.cache/huggingface
```

以后：

```bash
hf download ...
```

就不会再把几十 GB 写进 `/root/.cache/huggingface`。

---

## 19. 不要再这样操作

以后尽量不要：

```bash
sudo pip install ...
sudo uv pip install ...
sudo hf download ...
```

尤其如果你是在普通用户环境。

你的 Qwen 环境已经在：

```text
/mnt/data/txhan/qwen3-coder/.venv
```

使用：

```bash
source /mnt/data/txhan/qwen3-coder/.venv/bin/activate
uv pip install ...
```

即可。

---

## 20. 最后做一次空间审计

清完后：

```bash
df -h /
```

然后：

```bash
sudo du -xhs / 2>/dev/null
```

再：

```bash
sudo du -xhd1 /root 2>/dev/null | sort -h
```

以及：

```bash
sudo du -xhd2 /root 2>/dev/null | sort -h | tail -30
```

你的目标不是把 `/root` 清成零，而是从现在的：

```text
~642G
```

降到合理范围。

按照目前已知内容，如果：

- uv cache：释放 ~41G
- pip cache：释放 ~46G
- Conda cache：可能几十 G
- 不用的 Conda env：可能几十～上百 G
- Triton/vLLM/torch cache：若有则继续释放
- 那未知的 ~300G 找出来后再针对性处理

最终**很可能可以释放 150～300GB，甚至更多**，而不需要动正在下载的 DeepSeek。

最重要的一条原则是：**暂时保留 `/root/.cache/huggingface`，直到确认 DeepSeek 当前下载不依赖它。** 其它 `uv`、`pip` 缓存可以先清。
