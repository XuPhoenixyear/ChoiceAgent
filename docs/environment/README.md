# my_rec：ChoiceAgent 项目开发环境

创建日期：2026-10-20。此环境位于当前 Linux A6000 服务器。

- 名称：`my_rec`
- 路径：`/datas/wuxi/anaconda3/envs/my_rec`
- Python：3.11.17
- GPU 运行时：PyTorch 2.10.0 + CUDA 12.8 官方预编译轮子
- 模型栈：Transformers 4.57.6、Sentence Transformers 5.2.0、Accelerate 1.12.0
- 检索与数据：FAISS CPU、SQLite FTS5、NumPy、SciPy、pandas、scikit-learn、PyArrow
- 应用：FastAPI、Uvicorn、Pydantic、Streamlit、HTTPX、DeepSeek 兼容客户端
- 开发：pytest、pytest-asyncio、Matplotlib、IPython kernel

## 使用

```bash
source /datas/wuxi/anaconda3/etc/profile.d/conda.sh
conda activate my_rec
python --version
```

选择编辑器解释器时使用：

```text
/datas/wuxi/anaconda3/envs/my_rec/bin/python
```

环境激活后会设置 `PYTHONNOUSERSITE=1`，避免导入用户目录中其他项目的包；`TMPDIR` 指向数据盘，避免大型安装占满系统临时目录。GPU 编号不固化到环境中，实际训练/推理前按服务器分配选择一张卡。

## 依赖文件

- `environment.yml`：Conda 部分；不包含完整 pip 依赖。
- `conda-explicit-linux-64.txt`：当前 Conda 基础包的精确构建记录。
- `requirements.in`：直接依赖与版本范围。
- `constraints.txt`：后续加包时保持模型/数值栈稳定的约束。
- `requirements.lock.txt`：本次安装成功后导出的全部 pip 版本。
- `verification.json`：实际验证结果与版本。
- `verify_environment.py`：可重复的轻量环境检查。

新增依赖时推荐使用当前环境的 Python，并保留核心约束：

```bash
python -m pip install -c /datas/wuxi/Projects/WorkSpace2/ai-search-agent-project-design/environment/constraints.txt PACKAGE_NAME
python -m pip check
```

## 在同类 Linux 服务器复现

以下命令用于尚未存在同名环境的机器；当前服务器已经创建完成，不需要再次执行。修改数据盘临时目录以适配目标机器。官方 PyTorch 安装源用于先安装 CUDA 轮子，随后安装锁定依赖。

```bash
mkdir -p /datas/wuxi/.cache/my_rec/tmp
export TMPDIR=/datas/wuxi/.cache/my_rec/tmp
cd /datas/wuxi/Projects/WorkSpace2/ai-search-agent-project-design/environment
conda env create -f environment.yml
conda activate my_rec
python -m pip install 'torch==2.10.0' --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.lock.txt --index-url https://pypi.org/simple
python -m pip check
python verify_environment.py
```

锁文件针对 Linux x86_64 / Python 3.11，不直接用于 Windows。模型权重按后续实现需要单独下载，DeepSeek 密钥按启用 API 时配置。

## 验证范围

本次验证全部通过，`pip check` 未发现依赖冲突；详细版本和检查结果见 `verification.json`。

检查核心包导入、CUDA 小矩阵计算、随机初始化的微型 Qwen3 前向、FAISS 检索、SQLite FTS5/BM25 和 FastAPI/Pydantic 参数校验。验证脚本只在自己的短期进程中选择当前空闲显存最多的 GPU（若已指定 `CUDA_VISIBLE_DEVICES` 则尊重该设置），不进行训练、下载预训练权重或调用付费 API。

vLLM、Flash Attention、LoRA 与量化库不作为首版环境前置条件。若后续需要 vLLM，按其版本要求创建独立推理环境，避免它改写主开发环境的 Torch/Transformers 组合。

官方参考：[PyTorch 安装](https://pytorch.org/get-started/locally/)、[Qwen3 模型卡](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)、[Qwen Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)。
