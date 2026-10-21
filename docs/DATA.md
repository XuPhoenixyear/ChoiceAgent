# 数据、检索与复现说明

本文对应两周 MVP 的实际实现。论文提出的方法与后续六周版本，不作为本版本已实现的能力。

## 1. 官方来源与冻结目录

主数据为 McAuley Lab 的 **Amazon Reviews 2023 / Office_Products 商品元信息**：

- [官方数据说明与字段定义](https://amazon-reviews-2023.github.io/)
- [官方 Office_Products 元信息文件](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Office_Products.jsonl.gz)

获取脚本对官方 gzip 文件进行流式读取，取得前 3000 条有效且 `parent_asin` 唯一的记录后停止，不下载完整类目。它是**按源文件顺序的确定性前缀样本**，不是随机样本，也不代表整个类目的品类、价格或品牌分布。此次扫描 3000 行，得到 3000 个商品，无重复或损坏行。

冻结目录版本为 `amazon23-office-prefix-3000-v1`。采集时间、原始样本 SHA-256、清洗目录 SHA-256 和抽样说明见 `data/processed/catalog.manifest.json`；实际缺失率和子类目统计见 `reports/catalog_audit.json`。

| 项目 | 实际数量 |
| --- | ---: |
| 商品 | 3000 |
| 历史价格已知 | 1539 |
| 历史价格缺失 | 1461 |
| 品牌缺失 | 272 |
| 特征列表缺失 | 563 |

`data/raw/office_products_metadata_sample.jsonl` 保留上述商品的原始元信息；`data/processed/catalog.jsonl` 是在线服务使用的规范化目录。原始元信息里可能存在评分汇总等字段，**不代表在线服务使用这些字段**。

本版本不读取评论、不构造曝光或点击日志，也没有声称重建真实历史推荐环境。没有全量行为数据上的训练/验证/测试划分，没有线上 CTR、转化率或 GMV 结论。

## 2. 字段语义、证据与缺失值

- `item_id` 采用官方 `parent_asin`，不把不同变体的子 ASIN 当作独立父商品。
- `price_usd` 是**数据集采集时的美元价格**；不是实时价格，不换算人民币，不承诺当前有货。非数值、区间、负值与非有限值视为未知。
- `brand` 仅取 `details.Brand`。`store` 是商店信息，不能直接当作商品品牌；缺失品牌保持 `null`。
- 类目路径、标题、特征、描述与属性来自原始字段，进行空白规范化与有界截断，不由大模型补写。
- 清洗目录剔除 `average_rating`、`rating_number`、`bought_together`、`details.Best Sellers Rank`、`details.Customer Reviews` 等聚合信息，不用它们伪造历史预测特征。
- 缺失价格不能通过任何严格价格上限或下限；缺失属性在比较时返回 `null`，界面显示“未知”。没有数据证据时不推测尺寸、材质、兼容性或电池续航。

证据 ID 使用 `<item_id>:<field>`，例如 `ASIN:price_usd`、`ASIN:attributes.Material`、`ASIN:features.0`。每条证据保存精确字段值和原始 gzip 行定位。非空 `features` 另有聚合 `ASIN:features` 证据，供列表比较使用。比较结果的每个非空字段，都必须指向**同一商品卡片内、值相同**的证据；空特征列表显示未知，不生成事实证据。

原始数据与模型权重不会随代码仓库发布。项目代码的许可证不等于原始评论、商品元信息或图片的再分发授权；公开演示可使用处理脚本和本项目自制 fixture，分发原始样本前应核对数据方条款。

## 3. 索引、模型和版本校验

SQLite 目录位于 `runtime/catalog.sqlite3`。`CatalogRepository.initialize()` 校验 JSONL 的 SHA-256、唯一 ID 和统一目录版本，内容变化时重建商品表及 FTS5。在线读取每次使用独立只读连接；会话和反馈使用另一数据库，不写入目录。

精确语义索引位于 `artifacts/indexes/`：

| 文件 | 用途 |
| --- | --- |
| `catalog.faiss` | FAISS `IndexFlatIP`，3000 × 2560 维的精确内积索引 |
| `embeddings.npy` | 相同顺序的 float32 单位向量，供显式喜欢的内容聚合使用 |
| `item_ids.json` | 向量行号到商品 ID 的映射，按 ID 稳定排序 |
| `manifest.json` | 目录、ID、索引、向量校验和以及编码契约 |

语义模型读取已有本地 `/datas/wuxi/Projects/datas/models/Qwen3-Embedding-4B`。它是独立的编码模型；Agent 生成接口采用 `/datas/wuxi/Projects/datas/models/Qwen3-1.7B`。本模块不下载模型，不调用付费 API。

离线和在线统一使用 Transformers、last-token pooling、L2 归一化及 **384 token 截断**。文档编码文本依次包含标题、类目、类目路径、品牌、特征、描述和属性。查询保留原始用户语言，并加以下检索指令：

```text
Instruct: Given a web search query, retrieve relevant passages that answer the query
Query: <原始查询>
```

GPU 使用 BF16 模型参数，归一化后向量保存为 float32。离线默认 batch size 为 8；在线查询为单条。384 token 是资源折中，长商品描述可能被截断，因此缺失召回不等于商品不存在。

启动检查目录内容校验和、ID 集合与顺序文件校验和、FAISS/向量文件校验和、维度、单位向量、索引类型、pooling、查询指令和本地模型指纹。一项不一致时禁止使用该语义索引，显式降级为词面检索。模型配置与 tokenizer 文件使用 SHA-256；大权重文件记录大小和修改时间，**并未宣称对整份模型权重做密码学哈希**。本地模型目录按只读不可变资产使用；替换权重后应重建索引。

`/health` 的 `dense_ready` 表示索引与模型路径契约可用，并不表示生成服务已经就绪。编码模型在第一次语义查询时才加载；普通健康检查不会触发生成调用。

## 4. 两路召回与相关性保护

先对整个冻结目录执行硬约束：历史价格上下限、排除品牌、类目、必含词和排除词。必含/排除词基于规范化文本的确定性字面匹配，不是商品属性分类模型；未覆盖的同义词或否定表达可能需要用户换词。

中文界面与英文目录之间，词面路线使用代码内可审计的有限词表，例如“桌面收纳架 → desk organizer”。该词表不是通用翻译器；语义路线始终使用原始查询。对已覆盖的复合商品类型执行标题/类目锚点保护，并排除“Mouse Pad for Wireless Mouse”一类把适配设备写进标题的配件误匹配。长词面 OR 查询（不少于 3 个检索词）至少需要 2 个词在商品文本中出现，避免单个偶然命中。

### BM25 路线

SQLite FTS5 使用 `unicode61`，检索词经过安全 token 化并组成字面 OR 查询，不接受用户直接传入 FTS 或 SQL 语法。标题、正文、品牌、类目字段的 BM25 权重分别为 `5, 1, 2, 1`。SQLite BM25 **分数越小越好**，按升序排列；不直接与余弦分数相加。

扫描上限 5000，大于当前整个 3000 商品目录；硬约束和相关性保护后，每路线保留最多 200 个候选。

### 语义路线

单位向量的内积等于余弦相似度。对当前目录进行精确搜索，过滤不满足约束或类型保护的项后，使用：

```text
threshold = max(0.42, 当前合格候选最高余弦 - 0.12)
```

只保留达到阈值的最多 200 项。`0.42` 是开发样例上的保守拒绝门槛，**不是相关性概率，也不是经过正式标注集统计校准的阈值**。它可能拒绝部分正确结果；后续应在冻结的人工标注开发集上调参，再在独立测试集报告效果。

`mode=lexical` 使用词面路线，`mode=dense` 使用语义路线，`mode=hybrid` 使用两路。语义运行失败时，任何需要语义的模式均显式尝试词面回退；有卡片时报告 `degraded`，没有卡片时报告 `retrieval_incomplete`，同时保留故障警告，不静默调用其他模型或外部 API。

## 5. 确切融合与个性化公式

对两路候选的并集做秩融合。路线内秩 `r` 从 1 开始：

```text
base_score(i) = Σ(route ∈ 可用召回路线) 61 / (60 + rank_route(i))
```

只有商品在该路线真实出现时才有该项贡献。它是常见 RRF 的等比例缩放，便于设置有界的软偏好权重。结果卡片保存 `route_ranks` 和匹配原因，便于追溯路线贡献。

当 `personalize=true` 时，显式不喜欢的商品从候选中移除；喜欢仅在**已满足当前查询相关性和硬约束**的候选内部产生软调整，不把无关的历史商品强行召回。

有向量产物时，对合法喜欢商品的单位向量求均值并再次归一化：

```text
profile = normalize(mean(embedding(liked_item)))
preference(i) = max(0, dot(embedding(i), profile))
final_score(i) = base_score(i) + 0.15 × preference(i)
```

没有可用向量产物时，用同一检索 token 规则抽取的有界词集合（每文本最多 20 个去重词）计算喜欢集合并集与候选集合的 Jaccard 相似度，仍乘 `0.15`。没有合法喜欢项或均值范数为零时不加偏好分。得分相同按商品 ID 排序，保证重复请求结果稳定。`personalize=false` 同时关闭喜欢调整和不喜欢排除。

这些分数不是校准点击率、购买概率、竞价分数或收益预测。本版本实现的是**显式反馈驱动的内容个性化**，没有训练 ItemCF/MF/BPR、没有学习行为用户向量，也没有实现 SAILRec 或 PACERec 的完整模型。后续如果引入 CF，必须独立提供真实行为来源、时间切分和离线评测；不得将当前内容聚合称为“已完成协同过滤”。

## 6. 无匹配与证据边界

- `no_match`：完整冻结目录中，没有商品满足这组结构化硬条件，例如零美元预算。该结论只适用于当前冻结目录及已执行的规则。
- `retrieval_incomplete`：存在满足硬条件的商品，但召回与相关性保护没有找到可展示候选。不能声称整个目录没有符合用户意图的商品。
- `degraded`：语义故障但有词面候选，返回明确标记的词面结果。
- `ok`：当前选定模式完成并产生候选，不等于商品适用性已获人工验证。

商品事实卡片来自目录；语言模型不能凭记忆新增商品或覆盖字段。比较只能操作本会话已展示的 ID；缺失信息显示未知。状态隔离、请求幂等和证据对齐分别有自动化测试。

## 7. 完整重建步骤

在服务器执行，文件均写入项目根目录下。当前可直接使用已创建的 `my_rec` 环境：

```bash
source /datas/wuxi/anaconda3/etc/profile.d/conda.sh
conda activate my_rec
cd /datas/wuxi/Projects/WorkSpace-RA
export PYTHONNOUSERSITE=1

# 重新获取官方前缀；会替换当前原始样本与清洗目录。
python scripts/prepare_catalog.py --root /datas/wuxi/Projects/WorkSpace-RA --limit 3000

# SQLite 与语义索引一并重建；进程只可见物理 GPU 5。
CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
  python scripts/build_index.py \
  --root /datas/wuxi/Projects/WorkSpace-RA \
  --model /datas/wuxi/Projects/datas/models/Qwen3-Embedding-4B \
  --device cuda:0 --batch-size 8 --max-length 384
```

重建会重新读取官方数据；若上游文件变化，样本和内容校验和可能变化。对比新旧 manifest 后再更新演示与评测，不把变化后的目录混用旧索引。离线作业与在线服务不应同时重建/读取半发布的索引；重建完成后重启业务服务以载入新产物。项目总 GPU 并发上限为 2，默认编码与生成均共用 GPU 5；不使用无限制的 `device_map=auto`。

## 8. 已做验证与尚未做的评测

`reports/retrieval_gpu_probe.json` 记录 11 个开发冒烟样例，包括常见办公商品、严格预算、无意义词和域外请求；`reports/dense_score_calibration.json` 保存阈值选择时观察到的开发样例余弦。它们是调试证据，**不是独立人工标注检索基准**，不能据此报告 NDCG、Recall、准确率或泛化收益。

当前索引在 GPU 5 上构建用时 136.075 秒。一次开发探针首条查询包含冷加载约 6.585 秒；同次运行的几个暖态普通查询约 0.106—0.251 秒。这些是少量观测值，不是 P95、吞吐承诺或压力测试结果；精确环境和每条记录见报告。

测试用 `data/fixtures/catalog.jsonl` 是 3 个明确标记的**自制测试商品**，只用于隔离的 CPU 单测/HTTP 集成，不混入真实目录。自动化测试重点验证预算、ID、证据、会话可见性、并发幂等、错误契约和失效降级。正式搜索质量评测、推荐离线评测及跨用户泛化留待下一阶段单独完成。
