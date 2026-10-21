# 商品目录中文展示缓存

中文展示缓存独立于原始商品证据。源文件 `data/processed/catalog.jsonl` 不会被翻译脚本改写；API 继续保留原始标题、要点、属性、来源链接和原文定位。展示缓存为 `data/processed/catalog.zh-CN.jsonl`。

## 构建与恢复

在项目根目录与 `my_rec` 环境中运行：

```bash
python scripts/localize_catalog.py --trial-size 32 --batch-size 32
python scripts/localize_catalog.py --batch-size 48
```

脚本只读取本地 `/datas/wuxi/Projects/datas/models/Qwen3-1.7B`，温度为 0，关闭思考，离线加载，显式绑定物理 GPU 5。一个进程内批量推理，不调用外部 API，不启动额外推理 worker。共享 GPU 的负载会明显影响耗时。

`runtime/localization/cache.jsonl` 逐批追加并同步写盘；重新运行会复用相同 `item_id + source_hash + translation_version` 的结果。`runtime/localization/raw.jsonl` 保留原始生成文本及校验问题。不要把这些展示翻译作为额外训练标签或商品事实来源。

在构建过程中，可在另一个仅使用 CPU 的进程发布全部商品的展示层：

```bash
python scripts/localize_catalog.py --watch-cache
```

该命令不加载 Torch。它先发布已有翻译和其余商品的保守中文类目占位，之后每增加 240 条缓存，原子更新一次完整 3000 行文件。`runtime/localization/progress.json` 标明真实已缓存数量，不能把占位数量算作模型翻译成功数量。

## 内容与校验边界

每个商品最多摘译原始前三条要点，每条只向模型提供前 220 个字符。冗长型号清单和营销性文字可能被删减；不会为原始空要点生成要点。输出截断时，仅恢复已完整闭合的 JSON 字符串；英文标题和无效标题再进行一次独立的标题翻译。仍失败时，标题只保留原始品牌和可核验的源类别，并标为 `rule`。

标题及要点中的新增阿拉伯数字会被拒绝；每条要点的数字与其对应原始要点单独核对。属性完全来自原始字段的受控中文词典与单位翻译，不接受模型对品牌、制造商、型号等身份信息的改写。对源文明确包含的 `rhinestone`、`Letter/Legal`，应用受控词义修正，避免“钻石”和“法律文件”等误译。

这些检查不能证明机器翻译完全正确。原文仍是事实依据；品牌、型号、罕见术语和宣传性功效需人工核查。展示层必须明确标注“中文翻译”，并提供查看原文的入口。

## 哈希合同

`source_hash` 由 `choiceagent.localization.source_hash` 统一计算，覆盖原始 `title/category/category_path/brand/features[:3]/attributes`，使用 `ensure_ascii=False`、`sort_keys=True`、`separators=(',', ':')` 的 JSON 进行 SHA-256。价格、原始描述和 `item_id` 不在该哈希中。展示层只接受来源哈希和版本均匹配的缓存。

## 报告

- `reports/localization_trial_v0.json`：首次完整属性翻译试验，失败基准，不能作为最终覆盖率。
- `reports/localization_trial_v1.json`：缩短输出但尚未加入截断恢复和标题重译的试验。
- `reports/localization_trial.json`：最终 32 项开发样本试验，机器翻译成功不代表语义泛化正确率。
- `reports/localization_build.json`：全量构建配置、耗时、模型/规则覆盖率、生成问题、样例和限制。

## 失败标题二次审校与最终导出

少数 Qwen 失败标题由 AI 编码助手依据同一份源商品标题进行二次审校；这些结果明确标为 `coding-assistant-source-review`，不属于 Qwen 输出，也不称为人工审校。它们只替换原 Qwen 失败行的标题，不能覆盖已有成功译文，也不能通过这个入口新增要点或属性。

在全量缓存完成、二次审校补齐之后执行：

```bash
python scripts/localize_catalog.py --finalize-report
```

最终导出会重新校验来源哈希、版本、品牌之外的中文商品描述及新增阿拉伯数字，并持久化已接纳的审校条目到 `data/processed/catalog.zh-CN.review.jsonl`。后续可以直接读取此文件恢复二次审校，无需保留 `runtime` 目录。最终报告同时保留原 Qwen 的模型/规则数量，分别列出 Qwen 标题与 AI 助手审校标题的数量，并记录展示文件和审校文件的 SHA-256。

最终报告的机器检查不覆盖全部语义风险，例如中文数字改写、罕见术语及原文营销性功效。其结论仅表示通过已列出的结构和来源一致性检查；商品事实仍以原文为准。
