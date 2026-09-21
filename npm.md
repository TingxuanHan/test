# 使用用户目录安装Pi与HTTP 407代理排查

如果安装nvm时出现：

~~~text
Received HTTP code 407 from proxy after CONNECT
~~~

这不是nvm本身的问题，而是服务器访问GitHub时经过了一个需要认证的代理。

当前最省事的方案是：先不安装nvm，直接使用服务器现有的Node.js和npm，把Pi安装到当前用户目录。这样既可以绕开<code>/usr/local/lib/node_modules</code>的权限问题，也不依赖GitHub clone。

## 1. 检查现有Node.js和npm

~~~bash
node -v
npm -v
which node
which npm
~~~

确认Node.js和npm都能正常运行后再继续。

## 2. 把npm全局目录改到当前用户HOME

~~~bash
mkdir -p "$HOME/.local/npm"
npm config set prefix "$HOME/.local/npm"
~~~

把Pi的可执行目录加入PATH：

~~~bash
grep -qxF 'export PATH="$HOME/.local/npm/bin:$PATH"' ~/.bashrc || \
echo 'export PATH="$HOME/.local/npm/bin:$PATH"' >> ~/.bashrc

source ~/.bashrc
~~~

确认npm prefix：

~~~bash
npm config get prefix
~~~

这里必须显示类似：

~~~text
/home/你的用户名/.local/npm
~~~

不能再是：

~~~text
/usr/local
~~~

再检查PATH：

~~~bash
echo "$PATH"
~~~

输出中应该包含：

~~~text
/home/你的用户名/.local/npm/bin
~~~

## 3. 安装Pi

~~~bash
npm install -g --ignore-scripts @earendil-works/pi-coding-agent
~~~

安装完成后验证：

~~~bash
which pi
pi --version
~~~

理想的Pi路径类似：

~~~text
/home/你的用户名/.local/npm/bin/pi
~~~

这样就避开了系统级目录：

~~~text
/usr/local/lib/node_modules
~~~

不要使用<code>sudo npm install -g</code>或<code>sudo pi</code>来绕过权限问题。

## 4. npm prefix仍然是/usr/local时

如果执行<code>npm config set prefix</code>后仍然显示<code>/usr/local</code>，说明有其他配置覆盖了用户设置。依次检查：

~~~bash
cat ~/.npmrc 2>/dev/null
env | grep -Ei 'npm|prefix'
npm config list
~~~

也可以强制本次安装使用用户目录：

~~~bash
npm_config_prefix="$HOME/.local/npm" \
npm install -g --ignore-scripts @earendil-works/pi-coding-agent
~~~

然后验证：

~~~bash
export PATH="$HOME/.local/npm/bin:$PATH"
which pi
pi --version
~~~

## 5. 排查HTTP 407代理来源

先检查环境变量：

~~~bash
env | grep -i proxy
~~~

再检查Git全局代理：

~~~bash
git config --global --get http.proxy
git config --global --get https.proxy
~~~

最后检查npm代理：

~~~bash
npm config get proxy
npm config get https-proxy
~~~

如果看到类似下面的地址，说明当前服务器确实配置了代理：

~~~text
http://xxx.xxx.xxx.xxx:xxxx
~~~

### 5.1 代理属于误配置

如果确认该代理不需要使用，可以先在当前Shell中临时清除：

~~~bash
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset ALL_PROXY
unset all_proxy
~~~

再清除Git全局代理：

~~~bash
git config --global --unset http.proxy 2>/dev/null || true
git config --global --unset https.proxy 2>/dev/null || true
~~~

测试GitHub连通性：

~~~bash
curl -I https://raw.githubusercontent.com
~~~

如果不再出现407并且连接正常，可以重新尝试安装nvm。

### 5.2 服务器必须通过代理访问外网

如果清除代理后出现：

~~~text
connection timed out
~~~

或：

~~~text
could not connect
~~~

说明服务器网络必须通过代理访问外网。此时HTTP 407通常表示代理需要用户名、密码或其他认证。不要继续绕过，应向服务器或网络管理员获取正确的代理地址和凭据。

不要把包含用户名、密码或令牌的代理URL提交到Git仓库或粘贴到公开日志。

## 6. 当前目标的最短执行流程

当前目标只是让Pi连接本机的Qwen服务，因此可以先不处理nvm：

~~~text
已有Node.js和npm
        ↓
~/.local/npm
        ↓
Pi
        ↓
127.0.0.1:8001
        ↓
vLLM
        ↓
Qwen3-Coder
~~~

按顺序执行：

~~~bash
node -v
npm -v

mkdir -p "$HOME/.local/npm"
npm config set prefix "$HOME/.local/npm"

grep -qxF 'export PATH="$HOME/.local/npm/bin:$PATH"' ~/.bashrc || \
echo 'export PATH="$HOME/.local/npm/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

npm config get prefix

npm install -g --ignore-scripts @earendil-works/pi-coding-agent

which pi
pi --version
~~~

只要这组命令跑通，就可以继续配置Pi连接<code>http://127.0.0.1:8001/v1</code>。

如果<code>npm install</code>也返回HTTP 407，说明npm registry同样被代理拦截。下一步应专门修复Ubuntu服务器的代理配置，而不是继续修改Pi安装权限。
