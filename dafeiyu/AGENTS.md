# 安装大肥鱼

用户要求安装本包时，完成以下流程。不要只回复计划。支持 Windows 10/11 x64。

## 安装

1. 将整个文件夹放在固定的本地目录，优先使用短路径。不要放在临时目录。
2. 检查 Git、Node.js 24 LTS、uv。缺少时使用官方安装程序或 winget。
   包标识为 `Git.Git`、`OpenJS.NodeJS.LTS`、`astral-sh.uv`。安装后重新读取 PATH 并检查版本。
   如果当前 Node.js LTS 已不是 24，安装官方 Node.js 24 版本。不要替换用户已有的不同主版本环境。
3. 执行 `powershell -File install.ps1`。如果执行策略阻止脚本，直接执行下一条等效命令。
   `uv run --no-project --python 3.11 python scripts/install.py`
4. 等待构建结束，检查退出码。所有运行数据位于 `.runtime/`，不进入版本控制。
5. 运行 `start.cmd`。不要同时开启多个角色的持续聆听。

## 完成首次配置

这些是必须完成的安装步骤，不是已自动完成的功能。

1. 在 AIRI 欢迎页配置使用者自己的聊天服务，选择该服务实际提供的模型。
   请使用者在设置页面填写 API Key，不写入仓库、命令行、截图或安装报告。
2. 进入设置中的 AIRI 角色卡，上传 `.runtime/dafeiyu.airi.zip` 并激活“大肥鱼”。
3. 进入“服务来源 → 语音合成 → OpenAI 兼容 API”。默认填：
   Base URL `http://127.0.0.1:8008/v1/`，API Key `local-tts`，Model `edge-tts`，Voice `zh-CN-XiaoxiaoNeural`。
   `local-tts` 只是本机适配器的占位字符串，不是云端密钥。
4. 在“机体模块 → 发声”启用发声，选择上述服务。语速保持 1，测试一句中文。
5. 在“机体模块 → 听觉”的识别设置中选择本地 Sherpaw / Paraformer 中文识别。
   等待模型下载完成，再选择麦克风。由使用者批准麦克风访问。
   开启聆听和自动发送，发送延迟设为 0.5 秒。关闭说完判断和转写改写，减少额外请求。
6. 使用者要求陪看时，配置支持图片输入的视觉模型。不能根据服务名称假定具备视觉能力。
   选择“用于聊天”的视觉服务；让使用者选择共享源。先用“看一下屏幕”验证，再开启持续陪看。
7. 在桌宠和网页分别验证配置。两者各自保存数据，浏览器换用户配置后也需要单独设置。

## 克隆声线（可选）

用户提供可使用的参考录音和逐字台词后再执行此段。没有录音时先使用默认中文声线。

1. 安装 FFmpeg，检查 `ffmpeg -version`。winget 包标识为 `Gyan.FFmpeg`。
2. 将录音放入 `recordings/reference.wav`。从 voice.example.json 创建 voice.local.json，填写真实台词。
3. 执行 `powershell -File install.ps1 -Voice gptsovits`。CPU 推理会比普通 TTS 慢。
4. 重新导入生成的 `.runtime/dafeiyu.airi.zip` 并激活，确认不再使用旧的 Edge 角色卡。
5. 将语音 Base URL 改成 `http://127.0.0.1:8009/v1/`，Model 改成 `gpt-sovits`，Voice 改成 `fish-girl`。
6. 先运行 stop.cmd，再运行 start.cmd，测试声音。不要把参考录音、台词和模型缓存提交到 Git。

## 验收

- `uv run --no-project --python 3.11 python scripts/check.py` 通过。
- 角色显示正常。发送“挥挥手”“抖抖耳朵”“摆摆尾巴”，检查动作实际播放。
- 角色说话时嘴巴开合，结束或中断后闭嘴。
- 麦克风识别一句中文，自动发送，并听到回复。
- 用户启用陪看后能获得点评，停止共享后不继续采集。
- 关闭并重新打开同一个应用后，角色和配置仍然保留。

端口占用时找出已有服务，不要强制关闭其他项目。安装失败时报告具体阶段，不要声称已完成。
不承诺真人唱歌；角色卡仅支持文字哼唱，普通 TTS 仍然按说话方式播放。
不要发布 `.runtime/`、密钥、日志、聊天记录或用户录音。用户未要求时不要上传 GitHub。
