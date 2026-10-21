# 知选 ChoiceAgent

两周 MVP：在同一商品发现系统中实现**搜索推荐链路与工具型 Agent**。使用 3,000 件真实商品元数据、本地 Qwen3-1.7B、BM25 + 语义检索、显式偏好和可核验的比较结果。适合展示检索、排序、会话、模型服务、评测与故障处理的完整工程能力。

项目根目录：`/datas/wuxi/Projects/WorkSpace-RA`。Python 环境：`my_rec`。默认在线模型共用物理 GPU 5；中文适配构建期间，在线服务使用 GPU 1、离线翻译使用 GPU 5，**合计最多两张卡**。未调用付费 API。

代码仓库：<https://github.com/XuPhoenixyear/ChoiceAgent>。项目级 AI 开发约定见 [AGENTS.md](AGENTS.md)，已配置的工程、设计与测试技能见 [.agents/README.md](.agents/README.md)。模型、运行状态、原始研究附件和过程录屏保留在本机，不随代码仓库同步。

## 直接运行

数据、索引与环境已经在当前服务器准备好。先进入项目并激活环境：

```bash
cd /datas/wuxi/Projects/WorkSpace-RA
source /datas/wuxi/anaconda3/etc/profile.d/conda.sh
conda activate my_rec
export PYTHONNOUSERSITE=1
python scripts/start.py --gpu-ids 5
```

界面：<http://127.0.0.1:8501>；接口文档：<http://127.0.0.1:8000/docs>。启动器等待三个服务健康，进程记录在 `runtime/services.json`。已启动时不要重复执行，停止使用：

```bash
python scripts/start.py --stop
```

远程服务器的 `127.0.0.1` 是服务器本身。Windows 电脑可在本地终端执行 `ssh -L 8501:127.0.0.1:8501 <你的服务器登录地址>`，再用本地浏览器打开 <http://127.0.0.1:8501>。无需将服务端口开放到公网。

试试：“找桌面收纳，预算35美元” → 点击喜欢 → “比较前两个商品的尺寸” → “预算改成20美元”。搜索页左侧条件只控制普通搜索；聊天条件写在消息里。

## 已实现

- 官方 Amazon Reviews 2023 Office_Products 元数据前 3,000 个有效商品；清洗、缺失审计、SQLite FTS5、来源记录与哈希清单。
- 普通检索、中文受控改写、精确 FAISS 语义检索、RRF 融合；先执行价格、品牌、类别、关键词硬约束。
- 会话内喜欢/不感兴趣/收藏，内容向量偏好增强和排除；当前明确约束始终优先。幂等事件、版本号与会话隔离。
- 本地 Qwen3-1.7B 选择 `search_catalog`、`compare_items`、澄清或结束；最多 4 次模型调用、6 次工具调用、一次修复和 60 秒回合期限。
- 工具参数验证、可见商品检查、重复调用停止、上下文/队列限制、失败时返回真实检索卡片或明确降级提示。
- 商品事实由后端证据字段渲染；未知价格/规格保留未知，历史美元价格不当作实时售价。
- 商品卡片、聊天推荐、比较表、分类筛选默认中文；独立中文展示缓存与原始商品事实分开，英文原文和证据可展开核对。
- 中文网页、按需运行记录、真实浏览器录屏、20 个真实模型开发验收场景、CPU 自动化测试与质量检查。

## 目录规划

| 目录 | 职责 |
| --- | --- |
| `src/choiceagent/` | 数据、检索、会话、Agent、API、本地模型服务与界面 |
| `scripts/` | 数据准备、索引构建、启停、质量检查、真实评测、录屏 |
| `configs/`、`.env.example` | 设计配置说明和运行时环境变量示例；运行以环境变量为准 |
| `data/raw/`、`data/processed/` | 官方原始样本、清洗目录、数据清单 |
| `data/fixtures/` | 自制、可提交的测试数据；与真实评测严格区分 |
| `artifacts/indexes/` | 精确向量索引、向量、ID 映射与完整性清单 |
| `tests/unit/`、`tests/integration/` | 无网络、无 GPU 的单元和服务集成测试 |
| `reports/` | 数据审计、覆盖率、模型兼容性、真实开发验收与演示 |
| `docs/design/`、`docs/research/` | 原项目设计书与两篇论文的项目内归档 |
| `docs/` | 架构、数据、操作、接口、安全边界和验收说明 |
| `runtime/`、`logs/` | SQLite 会话、进程记录、缓存、浏览器、临时文件与日志；不提交 Git |
| `.github/workflows/` | CPU 质量检查配置；真实 GPU 评测单独按需运行 |

模型权重复用已有只读外部目录，Conda 环境保留在其标准安装目录；其余本项目文件均在本根目录。原设计书中的大规模 CF 训练、曝光日志、CTR/重排训练、广告竞价、线上 A/B、多租户部署属于后续版本；本版本的个性化是**内容偏好**，不冒称已完成 CF 或广告系统。

## 复现与验收

```bash
python scripts/check.py
python scripts/evaluate_mvp.py
```

第一条运行静态检查、类型检查、自动化测试、依赖一致性与构建；不加载 GPU。第二条要求三个服务已启动，使用真实目录和本地模型并保存 HTTP 请求/响应与 trace，不使用模拟模型。结果入口为 `docs/ACCEPTANCE.md` 和 `reports/mvp_evaluation.md`。

- [数据、重建与排序算法](docs/DATA.md)
- [架构与工程取舍](docs/ARCHITECTURE.md)
- [部署、故障处理、环境复现](docs/RUNBOOK.md)
- [中文界面说明](docs/UI_GUIDE.md)
- [商品中文适配与事实边界](docs/LOCALIZATION.md)
- [中文缓存构建与恢复](docs/catalog_localization.md)
- [本地模型服务](docs/model_service.md)
- [接口契约](docs/contracts.md)
- [真实演示视频](reports/demo/final/choiceagent_demo.mp4)
- [原项目设计书](docs/design/知选ChoiceAgent_项目设计书_2026-10-20.pdf)

20 例是可见且用于调试的开发验收集，不能解释为未见任务成功率。3,000 商品是源文件前缀样本，价格来自历史元数据；系统不验证当前库存、运费、商家描述真实性或实时价格。中文改写和预算解析只覆盖有测试的常见表达。1.7B 控制器仍可能选错关键词，未知/失败会明确呈现；不承诺生产 SLA。

中文展示已覆盖全部 3,000 件商品标题，另有 6,037 条中文短卖点；标题中 2,920 条由本地模型翻译，80 条失败项由 AI 助手依据源文再次审校。最新覆盖率、原文保留检查与真实接口记录见 `reports/localization/api_acceptance.json`，浏览器截图与交互记录见 `reports/localization/`。译文是辅助阅读资料，可能存在术语偏差；仅摘译至多前三条原始卖点，其他详情仍可查看英文原文。此前 `reports/demo/final/` 的视频和 `reports/mvp_evaluation.json` 记录适配前的 MVP 验收，不作为中文翻译质量证明。

完成缓存构建后，可复核中文展示和真实服务：

```bash
python scripts/verify_localization.py
```

该检查不重新构建检索索引，也不测量翻译语义准确率；会检查全部商品的源哈希、中文标题覆盖、原始目录一致性，并实际调用搜索、比较与本地 Agent。
