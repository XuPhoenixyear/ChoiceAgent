# 本地模型服务与复现

本服务只读取 `/datas/wuxi/Projects/datas/models/Qwen3-1.7B`，不下载权重，不访问付费 API。进程在导入 Torch 前限制 GPU 可见性，默认物理卡 5；显式 `CUDA_VISIBLE_DEVICES` 优先，其次 `CHOICEAGENT_GPU_IDS`，必须为 1—2 个不重复数字 ID。绝不向调用方的 mask 追加卡 5。模型只放到可见设备的 `cuda:0`，不使用 `device_map=auto`。只启动单进程；所有缓存和临时文件位于项目 `runtime/`。

## 启动和接口

在 `my_rec` 环境、项目根目录执行：

```bash
conda activate my_rec
CHOICEAGENT_MODEL_PRELOAD=1 python -m choiceagent.model_server
```

默认仅监听 `127.0.0.1:8001`。`CHOICEAGENT_MODEL_PRELOAD=1` 启动时加载；省略则首次生成时加载。第一次冷载可能受磁盘速度影响，应先用预加载确认 `/health` 返回 `ready: true`，再启动 Agent 评测。`GET /health` 本身不会加载权重。

`POST /v1/chat/completions` 接受 `messages`、`temperature`、`max_tokens`，可选 `request_id`。Qwen3 chat template 明确 `enable_thinking=False`。输入与最大输出合计不得超过 8192 token，输出最大 768 token；超限返回 `MODEL_CONTEXT_LIMIT`，不会偷偷截断消息。

同步客户端 `LocalModelClient.generate(messages, *, max_tokens, temperature, deadline)` 的 `deadline` 是 `time.monotonic()` 绝对值。`GenerationResult` 提供 `text`、`usage`、`model`、`elapsed_ms`、`finish_reason`。失败抛出带 `code` 的 `ModelClientError`。客户端无自动重试、外部模型切换或付费 fallback。

## 资源和超时边界

- GPU 生成串行，最多 2 个等待请求，超出返回 429 `MODEL_BUSY`。
- HTTP 请求并发上限 16；模型服务单进程、单 worker。
- 解码每步检查 monotonic deadline、HTTP 断开取消与关闭事件；取消后的请求会释放推理锁和队列槽位。
- GPU 预填充和当前 kernel 无法中途打断，所以 deadline 是中止请求的控制边界，并非硬实时毫秒保证。真实报告保留所测延迟。
- 首次模型权重载入是进程级资源初始化，不保证被请求 deadline 打断；生产演示建议预加载，避免在用户请求中冷载。
- 默认日志仅记录 token 数、耗时、异常类型；请求校验错误不返回原始 prompt。

## 单元测试

```bash
pytest tests/unit/test_model_client.py tests/unit/test_model_server.py -q
ruff check src/choiceagent/model_client.py src/choiceagent/model_server.py scripts/evaluate_model_compatibility.py scripts/validate_model_runtime.py
mypy src/choiceagent/model_client.py src/choiceagent/model_server.py
```

单测使用 fake engine / mock transport 隔离队列、取消、deadline、协议错误与敏感内容边界，不把 fake 结果算作真实模型表现。

## 真实模型评测

服务 ready 后运行当前主 Agent system prompt 的 20 例：

```bash
python scripts/evaluate_model_compatibility.py
```

输出 `reports/model_compatibility_production.json`。包含完整 system prompt、prompt SHA256、每条 messages、未经剪裁的 raw_text、usage 和耗时。评测只表示 JSON 结构、动作选择与必要参数兼容性，不能等同检索质量、最终事实正确率或端到端任务成功率。

探索期两份报告保留为 `model_compatibility_baseline.json` 和 `model_compatibility.json`；可按原提示复跑并写入不同文件：

```bash
python scripts/evaluate_model_compatibility.py --replay-report reports/model_compatibility_baseline.json --output reports/model_compatibility_baseline_replay.json
python scripts/evaluate_model_compatibility.py --replay-report reports/model_compatibility.json --output reports/model_compatibility_examples_replay.json
```

正式脚本统一要求比较维度非空且属于允许字段，因此早期 baseline 的“只检查商品 ID”宽松结果可能与正式重测分数不同；原始结果与原始 raw_text 均保留，当前验收以 production 报告及其 `evaluator_version` 为准。

探索测试发现：即使 20 条均产出合法动作，1.7B 模型也会在最终回答里照抄示例价格，甚至给空结果或未知价格编造金额。因此主 Agent 只让模型规划和复核动作，商品事实由目录证据与固定模板渲染。报告显式保留这些失败，不能对外宣称“Agent 100% 成功”。

在没有其他客户端使用生成器时验证运行保护（会短暂产生 6 并发请求）：

```bash
python scripts/validate_model_runtime.py
```

输出 `reports/model_runtime_validation.json`，检查真实模型输入超限、解码超时、超时后恢复、队列容量和并发拒绝。此脚本不会新增模型进程或选择其他 GPU。
