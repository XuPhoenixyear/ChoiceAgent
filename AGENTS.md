# ChoiceAgent 项目开发约定

## 项目事实与文档

- 当前实现为 Python 3.11、FastAPI、Streamlit、SQLite FTS5、FAISS 与本地 Qwen3。界面通过 HTTP 访问 API，不直接加载模型。
- 开始任务先读 `README.md`，再按涉及范围读取 `docs/ARCHITECTURE.md`、`docs/contracts.md`、`docs/RUNBOOK.md`、`docs/ACCEPTANCE.md`。中文展示另读 `docs/LOCALIZATION.md` 和 `docs/catalog_localization.md`。
- `docs/design/`、`docs/environment/` 是历史设计和环境归档；当前代码、根目录依赖快照和上述运行文档优先。
- 通用 Skill 中的 React、Next.js、Postgres、Poetry、npm 和端口示例不代表本项目技术选型，不据此迁移现有架构或重建环境。

## Skill 发现与使用

- 项目技能位于 `.agents/skills/<name>/SKILL.md`，来源和版本见 `.agents/skills.lock.json`，用途见 `.agents/README.md`。
- 开始开发任务读取 `.agents/skills/using-superpowers/SKILL.md`，按任务选择其余技能；不要每次把全部技能正文载入上下文。
- 上游的 `superpowers:<name>` 指本地同名目录；没有专门 Skill 工具时直接读取对应 `SKILL.md`。代理、计划、终端工具映射到当前会话实际提供的工具，不照抄其他产品的工具名或参数。
- 技能是工程方法参考。遵守系统规则、用户本次要求和本文件；用户已授权的范围内持续推进可逆工作，不因上游通用“Ask First”条目重复确认。
- 不因技能示例修改用户全局 Codex 设置、模型选择或插件配置。诊断技能只在相关诊断任务中使用，安装技能不授权读取其他会话或向外发送会话内容。

| 任务 | 优先技能 |
| --- | --- |
| 新功能、范围与设计取舍 | `brainstorming` → `writing-plans` |
| 按计划实现 | `subagent-driven-development` 或 `executing-plans`，按任务独立性选择 |
| 行为变更、故障修复 | `test-driven-development`、`systematic-debugging` |
| API、数据结构与工具契约 | `api-and-interface-design` |
| 架构决定和文档变更 | `documentation-and-adrs` |
| URL 接入、外部数据、会话和工具安全 | `security-and-hardening` |
| 商品卡片、比较表、中文交互和移动端布局 | `ui-ux-pro-max` |
| 真实浏览器交互和界面验证 | `webapp-testing` |
| 审查与完成验收 | `requesting-code-review`、`receiving-code-review`、`verification-before-completion` |

- 仅在适用且当前工具支持时启用并行代理和 Git worktree；不为小修改创建额外工程流程。审查应基于实际文件差异和验证结果。
- Git 辅助脚本和 SHA 审查模板要求有效仓库、HEAD 和对应提交。条件不满足时用 `runtime/` 下任务记录和文件差异审查，不编造 SHA、不运行依赖 Git 的脚本；用户授权版本管理后可以建立仓库。

## 现有环境与运行资源

- 复用 Conda `my_rec`，解释器为 `/datas/wuxi/anaconda3/envs/my_rec/bin/python`，设置 `PYTHONNOUSERSITE=1`。已有环境不重复创建。
- 临时文件放项目 `runtime/tmp`，设置 `TMPDIR` 指向该目录；不向服务器 `/tmp` 写入大型下载或构建文件。
- 模型权重位于已有外部只读目录，不自动下载模型、不替换为付费接口。
- 默认在线模型使用物理 GPU 5；中文适配构建的在线 GPU 1 / 离线 GPU 5 配置以运行文档和既有服务为准，总计最多两张卡。文档或 Skill 配置任务不启动模型。
- 复用已有服务；需要管理服务时使用 `scripts/start.py` 并先检查 `runtime/services.json` 和健康状态。API 8000、生成服务 8001、Streamlit 8501 均在回环地址。
- API 保持单 worker、生成串行，不重复启动模型、不误停其他项目进程，不删除或重置正在使用的 SQLite 会话状态。

## 业务与架构边界

- 普通搜索与 Agent 共用 `DiscoveryService.search`，显式预算等硬约束优先于偏好和语义相似度。
- 商品事实、中文展示和模型生成分层；事实来自后端证据，未知信息保持未知，历史美元价格不能称为实时售价。
- 保持模型调用次数、工具次数、回合时限、输入校验、失败降级和会话隔离等现有边界。变更这些边界时同步契约、测试与说明。
- 网页商品链接和电商接口接入是后续增量；实现时按现有服务边界增加适配层，并验证来源、抓取时间、商品匹配和 URL 访问边界。

## 界面与浏览器技能适配

- UI UX Pro Max 的本地查询脚本为 `.agents/skills/ui-ux-pro-max/scripts/search.py`，使用项目 Python。当前没有 Streamlit 专用栈数据，优先查询 `product`、`ux`、`style`、`color` 等通用域，不套用 React/Next.js 实现。
- 保留现有界面中文规范和原始证据入口。设计结论必须核对本产品与移动端场景，不直接接受搜索工具的默认结果。
- 浏览器复用 `runtime/browser`，设置 `PLAYWRIGHT_BROWSERS_PATH` 为其绝对路径。优先沿用 `scripts/capture_demo.py` 的无 GPU 浏览器配置和现有运行服务。
- `webapp-testing` 中的 npm、5173 和临时目录只是示例；不要用通用启动包装器重复拉起本项目服务。动态页面通过元素、状态或接口就绪条件等待。
- 仅在需要视觉伴侣时启动 Superpowers 可选预览，保持回环监听、项目内输出；不因安装技能自动启动预览或额外网络请求。

## 验证与版本管理

- 源码行为修改按影响范围运行相关测试；常规完整质量入口是项目 Python 执行 `scripts/check.py`，包括 Ruff、mypy、CPU pytest、覆盖率门槛、依赖检查和构建。
- 只有文档或 Skill 配置变化时，检查文件、引用、元数据和涉及的工具即可，不启动 GPU 验收或覆盖既有真实评测报告。
- `scripts/evaluate_mvp.py`、`scripts/verify_localization.py` 需要真实服务；有相关行为变更时按文档运行。模拟测试、历史开发验收和新实测结果要明确区分，不把历史结果声称为本次验证。
- GitHub 仓库为 `https://github.com/XuPhoenixyear/ChoiceAgent`，沿用远端默认分支与历史；不强推，不擅自重写历史。
- 提交源码、测试、配置、项目文档和 `.agents/`。不提交密钥、运行会话、浏览器、模型、原始目录数据、生成索引、缓存和过程日志；示例环境文件仅保留占位值。
