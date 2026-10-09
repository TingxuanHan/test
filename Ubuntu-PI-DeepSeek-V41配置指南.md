# Ubuntu PI 配置自部署 DeepSeek V4.1

在已经安装好 Ubuntu 版 Pi Coding Agent（PI）的基础上，下面配置自部署的 `deepseek_V41`。

根据之前提供的信息，配置如下：

| 配置项 | 内容 |
| --- | --- |
| Provider 名称 | `deepseek-local` |
| 模型 ID | `deepseek_V41` |
| API 地址 | `http://10.90.79.129:8048` |
| API 协议 | Anthropic Messages（优先尝试） |
| 认证 Token | `ANTHROPIC_AUTH_TOKEN` |

建议单独创建 `deepseek-local` Provider，不覆盖 PI 自带的 Anthropic Provider，这样后续还能切换其他模型。

PI 官方文档确认，自定义模型配置文件的位置是 `~/.pi/agent/models.json`。参见 [PI 模型配置文档](https://pi.dev/docs/latest/models)。

## 第一步：创建配置目录

在 Ubuntu 终端执行：

```bash
mkdir -p ~/.pi/agent
nano ~/.pi/agent/models.json
```

如果之前已经配置过其他模型，需要保留现有的 Provider 配置，不要直接覆盖整个文件。

## 第二步：写入 DeepSeek 配置

将以下内容粘贴到 `models.json`：

```json
{
  "providers": {
    "deepseek-local": {
      "baseUrl": "http://10.90.79.129:8048",
      "api": "anthropic-messages",
      "apiKey": "$ANTHROPIC_AUTH_TOKEN",
      "authHeader": true,
      "models": [
        {
          "id": "deepseek_V41",
          "name": "DeepSeek V4.1 (Local)",
          "reasoning": false,
          "input": ["text"]
        }
      ]
    }
  }
}
```

这份配置使用 PI 官方支持的环境变量引用和 `authHeader`，用于向服务端发送 Bearer Token。参见 [PI 自定义模型配置说明](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/models.md)。

其中 `reasoning: false` 是初次连接的兼容性设置，暂时不要求网关支持 Anthropic 扩展思考参数。

保存文件：

- `Ctrl + O`，回车确认。
- `Ctrl + X`，退出 nano。

## 第三步：配置 API Token

在 Ubuntu 终端执行：

```bash
read -rsp "请输入 API Token: " ANTHROPIC_AUTH_TOKEN
echo
export ANTHROPIC_AUTH_TOKEN
```

输入原来配置中的完整 `sk-...` Token，回车即可。使用这种方式不会把 Token 明文写入 shell 命令历史。

再设置内网地址绕过代理：

```bash
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}10.90.79.129"
export no_proxy="$NO_PROXY"
```

注意：以上环境变量仅对当前终端及其启动的进程生效。

## 第四步：启动 PI

先检查模型是否加载成功：

```bash
pi --list-models deepseek_V41
```

再检查认证是否就绪：

```bash
pi auth check --provider deepseek-local
```

最后启动：

```bash
pi --provider deepseek-local --model deepseek_V41
```

也可以直接进行一次简单测试：

```bash
pi --provider deepseek-local --model deepseek_V41 \
  --print "请只回复 OK"
```

这些模型选择、认证检查和单次运行参数均在 PI 当前 CLI 文档中支持。参见 [PI 命令行文档](https://pi.dev/docs/latest/cli)。

## 第五步：如果出现报错

| 错误 | 处理方式 |
| --- | --- |
| `401 Unauthorized` | 检查 Token 是否完整及网关认证方式。 |
| `404 Not Found` | 确认网关是否支持 `/v1/messages`。 |
| `Connection refused` | 确认 Ubuntu 可以访问 `10.90.79.129:8048`。 |
| 模型不显示 | 检查 JSON 格式、环境变量和模型 ID。 |
| `400` 提示不支持工具参数 | 检查 Anthropic 兼容性配置。 |

如果提示不支持 `eager_input_streaming`，可以在 Provider 中增加：

```json
"compat": {
  "supportsEagerToolInputStreaming": false
}
```

如果网关只支持 OpenAI Chat Completions，则需要将 `api` 改为 `openai-completions`，并根据实际路由调整 `baseUrl`，通常为 `/v1`。不要在未确认协议前盲目修改。

建议先执行第二步到第四步。如果执行 `pi --provider deepseek-local --model deepseek_V41` 后报错，记录错误信息并隐藏 Token，再定位具体配置项。
