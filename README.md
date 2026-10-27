# 大肥鱼 · 微信里的 DeepSeek 聊天搭子

一个只用来聊天的微信 bot：走微信官方 ClawBot 插件同款的 iLink 协议，模型用 DeepSeek V4 Flash，带分层长期记忆，会在合适的时候主动找你。

整个项目约 1500 行 Python，依赖只有 `httpx` 和 `qrcode`，一台 2 核 2G 的云服务器就能跑。

---

## 一、整体结构

```
微信（你） ⇄ iLink 服务器（ilinkai.weixin.qq.com） ⇄ 大肥鱼（你的服务器） ⇄ DeepSeek API
                                                         │
                                                         └─ data/ （SQLite + 画像文件）
```

| 文件 | 作用 |
|---|---|
| `dafeiyu/ilink.py` | iLink 协议客户端：扫码登录、长轮询收消息、发消息、"对方正在输入" |
| `dafeiyu/llm.py` | DeepSeek 调用（关闭思考模式，统计缓存命中） |
| `dafeiyu/memory.py` | 记忆系统和 prompt 拼装，**省 token 的关键都在这里** |
| `dafeiyu/proactive.py` | 主动发消息的判断规则 |
| `dafeiyu/web.py` | 联网搜索（DeepSeek Search API），以工具的形式交给模型自己决定要不要查 |
| `dafeiyu/bot.py` | 主循环：收消息、攒消息、分条发送、后台整理记忆 |
| `persona.example.md` | 人设模板，复制成 `persona.md` 再改（`persona.md` 不进仓库） |
| `data/profile.md` | 核心画像，自动生成，你也可以手改 |

协议细节对照的是腾讯官方插件 `@tencent-weixin/openclaw-weixin@2.4.10` 的源码（请求头、`base_info`、`context_token`、错误码处理都与官方插件一致）。

---

## 二、记忆怎么设计的

记忆分三层，越往上越精炼、越常用：

1. **核心画像** `data/profile.md`：他是谁、做什么、在意什么。每次对话都带上，上限 1500 字。每天凌晨 4 点由模型根据新记忆整理一次，旧版本备份在 `data/profile_history/`。
2. **零碎记忆**（SQLite `notes` 表）：聊天中提炼出来的具体事实，比如"他下周三去上海面试"。聊天停下 10 分钟后自动提炼。每次只按当前话题检索最相关的 6 条带上，不会全部塞进去。
3. **对话摘要**：上下文超过 40 条消息时，把最早的 20 条压缩进摘要。

另外有一张 `followups` 表，记"之后要问问他"的事（比如面试结果）。提炼时由模型顺手生成，到点后主动消息会用到。

几个刻意的取舍：

- **只记你说过的事实**，不做性格分析、不下心理结论。你说"忘了吧"，相关记忆会被作废。
- **检索不用 embedding**，用中文二元组重合度打分。DeepSeek 没有 embedding 接口，个人记忆量（几百到几千条）用这个方法足够，还省掉一个依赖和一份费用。
- 所有记忆都是本地明文：画像是 Markdown，其余在 SQLite 里，随时能看、能改、能删。

## 三、为什么不费 token

DeepSeek 会自动缓存请求前缀。V4 Flash 缓存命中的输入价格大约是未命中的 1/50。所以 prompt 严格按"越稳定越靠前"的顺序拼：

```
[system]   人设 + 核心画像 + 对话摘要 + 聊天规则     ← 一天最多变一两次
[history]  最近的原始对话，只往后追加               ← 每轮只多出最后几条
[最后一条]  他的新消息 + 【系统备注】当前时间、检索到的相关记忆   ← 每次都变，放在最后
```

为此做的几个处理：

- 动态内容（当前时间、检索结果）只拼在**最后一条**消息后面，**不存进历史**，所以下一轮的前缀和上一轮完全一致。
- 历史压缩会改动 system prompt，所以只在聊天空闲时做，不会在聊到一半时让缓存失效。
- 聊天时关闭思考模式（`thinking: disabled`），否则思考内容按输出计费。

发送 `/状态` 可以看今天的 token 用量和缓存命中率。正常聊天时，大部分输入 token 都应该命中缓存。

粗略估算：每天聊 100 来条，加上记忆整理和主动消息的调用，一个月在几块钱这个量级（按 V4 Flash 官方价）。

## 四、像真人的细节

- **攒消息**：你连着发好几条，它会等你停下 1 秒再一起回。
- **看完再回**：生成回复期间你又发了新消息，它会放弃刚写好的回复，读完新消息重新组织。
- **分条发**：模型可以把一次回复拆成最多 3 条（空行也算分条），每条之间按字数停顿，同时显示"对方正在输入"。生成回复和"正在输入"同时进行，不额外拖慢。
- **会看时间**：知道现在几点、离上次聊天隔了多久。
- **主动找你**：每隔 25～70 分钟（随机）检查一次，满足条件后由模型判断要不要发、发什么。条件包括：在 9:00–23:00 之间、已经安静了 3 小时、你没有冷落它上一次主动发的消息、今天主动次数没超过 3 次。如果有惦记着要问的事，安静半小时就可以开口。模型觉得没必要时会回 `<PASS>`，这次就不发。主动消息一次最多连发 5 条，以分享它自己的念头为主，并且会看到自己之前主动说过什么，避免重复；连发途中你一开口，剩下的就不发了，先回你。

## 五、联网

模型有一个 `web_search` 工具，背后是 DeepSeek 的 Search API（`POST /v1/tools/search`，和聊天共用同一个 API Key）。

- **聊天时**：问到天气、新闻、游戏版本、价格这类实时信息，或者它拿不准的事实，它会先查再回；闲聊不查。一次回复最多查 2 轮。
- **主动找你时**：可以围绕你的兴趣找点最近的新东西来分享，不是每次都查。
- 查到的内容只在这一次请求里用，不存进聊天记录，所以不影响缓存命中。
- 价格约 $0.75 / 千次成功请求（空结果和失败不收费）；`/状态` 里能看到今天搜了几次。不想联网就把 `[web] enabled` 设成 `false`。
- 注意：`deepseek-flash` 不支持 DeepSeek 内置的 `web_search`（只在 `deepseek-v4-pro` 上生效），所以这里用的是独立的 Search 接口 + 普通的 function calling。

## 六、平台限制（iLink 规则，绕不过去）

- **24 小时窗口**：你 24 小时内没给它发过消息的话，它发的消息会被平台静默丢弃。所以主动消息只在你上次说话后 22 小时内触发。
- **回复额度**：你每发一条消息，它最多能回大约 10 条。代码按 9 条控制，额度快用完时会把剩下的内容合并成一条发。
- **频率**：README 早期按"5 分钟约 7 条"估计，但实测正常聊天远超这个数也没被拒。现在回复不预先限速，只有被拒时（`ret=-2`）等 30 秒重试一次；主动消息仍按 5 分钟 6 条控制，快超时剩下的直接不发。
- **独占**：同一个 bot 同时只能有一个程序在收消息。如果你之前用 OpenClaw 绑过 ClawBot，跑这个之前先把 OpenClaw 停掉。
- 不能拉进群，也不能被别人加好友，只有你自己能和它聊。

---

## 七、部署（以阿里云 / 腾讯云 Ubuntu 为例）

需要 Python 3.10 及以上。地域选国内就行，不需要备案（程序只往外发请求，不对外开端口）。

```bash
# 1. 放代码
sudo useradd -m -s /bin/bash dafeiyu
sudo mkdir -p /opt/dafeiyu && sudo chown dafeiyu: /opt/dafeiyu
# 把这个项目的文件传到 /opt/dafeiyu，然后：
sudo -iu dafeiyu
cd /opt/dafeiyu
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple

# 2. 配置
cp config.example.toml config.toml
nano config.toml          # 填 [llm] api_key（去 platform.deepseek.com 创建）
cp persona.example.md persona.md
nano persona.md           # 改人设（可选）

# 3. 绑定微信：手机微信先在「我 → 设置 → 插件」里启用「微信 ClawBot」
.venv/bin/python -m dafeiyu login     # 终端会显示二维码，用微信扫

# 4. 先在前台试跑，给 ClawBot 发条消息看看
.venv/bin/python -m dafeiyu run
```

试跑没问题后，按 Ctrl+C 停掉，改成 systemd 常驻：

```bash
exit   # 回到有 sudo 的用户
sudo cp /opt/dafeiyu/deploy/dafeiyu.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now dafeiyu
journalctl -u dafeiyu -f      # 看日志
```

## 八、日常使用

| 在微信里发 | 作用 |
|---|---|
| `/状态` | 今天的 token 用量、缓存命中率、剩余回复额度 |
| `/记忆` | 看它整理出来的核心画像 |
| `/ping` | 测试是否在线 |

- 想让它记住什么，直接说"记住……"就行。想让它忘掉，说"把……忘了吧"。
- 想直接改它对你的印象：编辑 `data/profile.md`，改完立刻生效。
- 调人设时，用 `python -m dafeiyu prompt` 可以看到最终发给模型的 system prompt。
- **备份**整个 `data/` 目录就是备份它的全部记忆。注意 `data/account.json` 是微信 bot 凭据，不要泄露。

## 九、常见问题

**提示登录态失效（ret=-14）**：重新运行 `python -m dafeiyu login` 扫码，然后重启服务。

**它不主动找我**：先确认最近 22 小时内你和它说过话，再确认当前在 `active_start`–`active_end` 时段内。用 `-v` 参数启动可以看到每次判断的原因。

**回复太长 / 太短 / 语气不对**：改 `persona.md`。想换风格时，直接在 prompt 里给两三句它"应该怎么说"的例句，效果最好。

**想换更聪明的模型**：`chat_model` 改成 `deepseek-v4-pro`。贵一些，但缓存命中的部分依然很便宜。

## 十、可以继续加的

- 图片理解：DeepSeek 是纯文本模型，现在收到图片只会显示成"[图片]"。可以加一个多模态模型，先把图片转成文字描述。
- 语音回复：iLink 支持发语音（silk 格式），可以接一个 TTS。
- 记忆检索升级：记忆量上万条以后，可以换成向量检索。
