# 运行与复现

## 环境

当前 `/datas/wuxi/anaconda3/envs/my_rec` 已安装项目可编辑包及开发依赖。始终设置 `PYTHONNOUSERSITE=1`，避免用户级旧包覆盖；项目临时文件必须位于 `runtime/tmp`，服务器 `/tmp` 空间较小。

`environment.yml` 记录 Conda 基础层；`requirements.lock.txt` 是本次实际安装快照。复建环境时先安装官方 CUDA 12.8 PyTorch，再安装锁定依赖与项目：

```bash
cd /datas/wuxi/Projects/WorkSpace-RA
conda env create -f environment.yml
conda activate my_rec
export PYTHONNOUSERSITE=1
export TMPDIR="$PWD/runtime/tmp"
mkdir -p "$TMPDIR"
python -m pip install torch==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.lock.txt
python -m pip install --no-deps -e .
python -m pip check
```

这是当前 Linux x86_64 / A6000 / 驱动595.91.07 的安装快照，不承诺跨平台二进制兼容。开发环境的额外包也在快照中。当前已存在的 `my_rec` 不应重复创建。`docs/environment/` 是设计阶段环境归档，最新依赖以根目录快照为准。

## 正常启停

```bash
python scripts/start.py --gpu-ids 5
python scripts/start.py --stop
```

启动器只接受 1—2 个不重复数字 ID，环境在任何 Torch 导入前生效，实际模型仅放在可见 `cuda:0`。使用 `0,1` 不代表自动双卡并行。当前实际用一张物理卡5。界面不导入 Torch，API 单 worker；生成服务单进程，最多1个生成与2个等待。

`runtime/services.json` 保存本次 PID 和命令标识，停止时验证进程命令，避免误杀重用 PID。不要另开 `uvicorn --workers 2`，不要同时启动两套模型。端口8000/8001/8501占用时，启动器报错，应确认哪个已有服务正在用端口，不杀其他项目进程。

运行配置使用 `CHOICEAGENT_*` 环境变量，见 `.env.example`。`configs/default.yaml` 是可阅读的设计快照，不参与配置加载。生成权重固定为用户指定的 `/datas/wuxi/Projects/datas/models/Qwen3-1.7B`；语义编码器复用已有 Qwen3-Embedding-4B。两者均只读本地权重，无自动下载或付费后备。

## 健康和故障

| 现象 | 处理 |
| --- | --- |
| 界面连接失败 | 检查 `logs/api.log` 与 `/health`，确认SSH转发与三服务是否运行 |
| 模型 `ready:false` | 等待预加载；检查 `logs/model.log`，目录是否存在、GPU是否有足够显存 |
| `MODEL_BUSY` / 429 | 当前串行队列已满；稍后用同 request_id 重试 |
| `SESSION_BUSY` / 409 | 同会话另一条聊天未完成，等待；不要绕过幂等 ID |
| `IDEMPOTENCY_CONFLICT` /409 | 同一 ID 对应不同内容；新内容必须使用新 ID |
| 语义索引失效/不可读 | 界面明确降级；按 DATA.md 重建，勿手工替换其中单个文件 |
| 无匹配与检索不完整 | 前者表示完整结构化可行集为空；后者只说明当前检索没有足够相关结果 |
| GPU OOM / 生成超时 | 检查仅有一套本项目服务；保留 trace，普通搜索仍可使用；别清理其他人的进程 |

生成 deadline 每解码步检查，但正在执行的 GPU kernel 和冷载不能硬实时抢占；因此超时不保证在指定毫秒精确中断。预加载后再评测，不能把冷载和热请求混成同一延迟指标。

日志在 `logs/`，模型运行初始化日志也可能在 `runtime/logs/`。请求日志含路径、状态与耗时，不含正文。traces、会话和收据位于 `runtime/state.sqlite3`。需要全新演示状态时，先停止三个服务，将 `state.sqlite3`、`state.sqlite3-wal`、`state.sqlite3-shm` 作为一组备份后再移走；不要在运行中单独删除 SQLite 文件。商品库 `runtime/catalog.sqlite3` 可从 JSONL 重建。

## 质量检查与真实评测

`python scripts/check.py` 运行格式、lint、类型、CPU测试/覆盖率、依赖检查和wheel构建，完整输出保存在 `reports/quality/`。GitHub Actions 使用不含模型的CPU测试依赖；尚未声称远程CI已执行。GPU测试不进入常规CI。

三个服务 ready 后：

```bash
python scripts/evaluate_mvp.py
python scripts/evaluate_model_compatibility.py
python scripts/validate_model_runtime.py
```

最后一条会施加并发队列负载，应避免与交互演示同时运行。20场景与动作兼容性是不同评测，均属于开发集。浏览器截图与视频见 `docs/UI_GUIDE.md`。报告包含历史时间戳，不因重启服务自动更新。
