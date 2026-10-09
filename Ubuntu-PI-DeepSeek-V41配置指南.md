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

## NotFound Error 排查

`NotFound Error` 通常意味着 Pi 请求的资源或 API 路径不存在，但还不能直接确定是模型 ID 错误，还是 API 协议不匹配。

目前的配置是：

- API 地址：`http://10.90.79.129:8048`。
- 模型 ID：`deepseek_V41`。
- 协议：`anthropic-messages`。

首先检查 PI 使用的 Anthropic Messages 路径是否与自部署网关实际提供的接口一致。

先不要重装 PI，也不要直接修改模型名。

### 第一步：检查 PI 是否识别了模型

在 Ubuntu 终端运行：

```bash
pi --list-models deepseek_V41
```

然后：

```bash
pi auth check --provider deepseek-local
```

正常情况下，第一条应显示自定义模型，第二条应显示 `ready`。

如果两条都正常，说明模型注册和本地认证配置基本就绪，但还不代表服务端 API 能成功响应。参见 [PI 命令行文档](https://pi.dev/docs/latest/cli)。

### 第二步：直接测试 API

这是判断 API 路径和协议是否匹配的关键一步。

先确保当前终端有 Token。如果没有，执行：

```bash
read -rsp "API Token: " ANTHROPIC_AUTH_TOKEN
echo
export ANTHROPIC_AUTH_TOKEN
```

然后执行：

```bash
curl --noproxy 10.90.79.129 -i --max-time 20 \
  http://10.90.79.129:8048/v1/messages \
  -H "Authorization: Bearer $ANTHROPIC_AUTH_TOKEN" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek_V41",
    "max_tokens": 64,
    "messages": [
      {"role": "user", "content": "Reply OK"}
    ]
  }'
```

看输出中的 HTTP 状态码：

| 结果 | 含义及下一步 |
| --- | --- |
| `200 OK` | Anthropic 接口可以响应，继续检查 PI 请求方式。 |
| `404 Not Found` | 重点检查 API 路径和模型 ID。 |
| `401 Unauthorized` | 检查 Token 或认证 Header。 |
| `400 Bad Request` | 检查请求参数及网关兼容性。 |
| `Connection refused` | 检查服务是否运行、端口及网络。 |

### 第三步：如果返回 404

进一步判断服务器是否提供 OpenAI 兼容接口：

```bash
curl --noproxy 10.90.79.129 -i --max-time 20 \
  http://10.90.79.129:8048/v1/chat/completions \
  -H "Authorization: Bearer $ANTHROPIC_AUTH_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek_V41",
    "messages": [
      {"role": "user", "content": "Reply OK"}
    ],
    "max_tokens": 64
  }'
```

如果第二个接口返回 `200 OK`，就把 `~/.pi/agent/models.json` 中 `deepseek-local` 的配置改为：

```json
"baseUrl": "http://10.90.79.129:8048/v1",
"api": "openai-completions"
```

其他模型配置先保持不变。Pi 官方支持这两类兼容接口，但配置必须与服务器实际提供的协议一致。参见 [PI 模型配置文档](https://pi.dev/docs/latest/models)。

继续定位时，需要提供以下两个信息：

1. 第一步 `pi --list-models deepseek_V41` 的输出。
2. 第二步 `curl` 的 HTTP 状态码和错误内容（隐藏任何密钥）。

这些信息用于进一步判断 API 地址是否需要调整，或 `deepseek_V41` 的模型路由是否存在。
