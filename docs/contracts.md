# ChoiceAgent MVP implementation contract

Python3.11, src layout. Root is /datas/wuxi/Projects/WorkSpace-RA. All app/runtime/data/reports go here. External read-only model paths and existing my_rec environment are exceptions. GPU process visibility must be <=2; use physical GPU5 by default for both model workloads (one GPU total). Never select device_map=auto outside an explicit visibility limit.

## Product JSONL (data/processed/catalog.jsonl)
One row: item_id(str parent_asin), title(str), category(str e.g. Office Products), category_path(list[str]), brand(str|null), price_usd(float|null), features(list[str]), description(str), attributes(dict[str,str]), source_url(str), source_locator(str), catalog_version(str).
Use original metadata facts, no aggregate ratings/counts. Price historical USD. Missing stays null. Evidence ID deterministic: `<item_id>:<field>`; fields title, price_usd, brand, category, features, features.N, attributes.KEY. The aggregate features evidence exists only for a nonempty source feature list.

## Core HTTP API (:8000)
All requests/response JSON. Session ID UUID minted by POST /v1/sessions -> {session_id,profile_version:0}. This is a local single-user demo, random session token isolates sessions; no arbitrary user IDs.
POST /v1/search {session_id,query,constraints:{max_price_usd:null,min_price_usd:null,excluded_brands:[],category:null,required_terms:[],excluded_terms:[]},limit:5,mode:'hybrid'|'lexical'|'dense',personalize:true} -> {trace_id,session_id,catalog_version,status:'ok'|'no_match'|'retrieval_incomplete'|'degraded',items:[{item_id,title,category,brand,price_usd,features,attributes,display,rank,score,route_ranks:{},reason_codes:[],evidence:[{evidence_id,item_id,field,value,source_locator}],missing_attributes:[]}],applied_constraints,profile_version,warnings:[],timing_ms:{},total_eligible:int}.
POST /v1/compare {session_id,item_ids:[2..4],dimensions:['price_usd','brand',...]} -> {trace_id,items:[...same cards],rows:[{dimension,label_zh,values:[{item_id,value,display_value,evidence_id:null|string}]}],warnings:[]} . Only session-visible IDs. `dimension` and `value` retain the source field identifier and original fact; `label_zh` and `display_value` are presentation aids. Unknown source values return value=null, display_value=null and evidence_id=null.
POST /v1/feedback {session_id,event_id(UUID),item_id,action:'like'|'dislike'|'save'|'remove'} -> {event_id,profile_version,liked_item_ids,disliked_item_ids,shortlist_item_ids} idempotent; only session-visible items.
GET /v1/sessions/{id} -> {session_id,profile_version,liked_item_ids,disliked_item_ids,shortlist_item_ids,visible_item_ids,last_result_order,constraints,messages}.
POST /v1/chat {session_id,request_id(UUID),message} -> {trace_id,session_id,request_id,status:'completed'|'clarification'|'degraded'|'no_match'|'retrieval_incomplete',answer,items:[],comparison:null|{rows:[]},tool_calls:[],warnings:[],usage:{},timing_ms:{}}.
GET /v1/traces/{trace_id}?session_id=... -> trace for same session only.
GET /health -> health without loading generator, version + catalog_count + dense_ready.
GET /v1/catalog/stats -> count, priced_count, categories, category_labels, brands, catalog_version, source. `categories` maps original category strings to counts; `category_labels` maps those same original strings to Chinese labels. Clients display labels but submit original category strings in constraints.
API errors: {error:{code,message,trace_id}}; 422 validation; 404 session/unknown item; 409 conflict; 429 busy; 503 unavailable. Strict bounded lengths. No implicit paid API.

## Chinese presentation and translation sidecar

Each generated product card includes `display` with this shape:

```json
{
  "locale": "zh-CN",
  "title": "中文商品标题",
  "category": "办公用品",
  "features": ["中文功能描述"],
  "attributes": {"材质": "金属", "商品型号": "A-12"},
  "translation_kind": "model",
  "source_language": "en"
}
```

`translation_kind` is `model` when a valid source-bound sidecar title is used, or `rule` for controlled fallback. A rule sidecar row uses the runtime fallback's known product-type label rather than replacing it with a generic brand/category placeholder. This kind is not a quality score or a guarantee that every source field has been translated. The response schema permits display=null for compatibility, while the current card builder populates it. Display attributes have Chinese field labels; the original card attributes keep original keys and values. Unknown fields/values may remain visible in the original data rather than being guessed. Missing original features remain missing; translated features never exceed min(3, source feature count). Known unit words can be rendered in Chinese, but amounts and physical units are not converted. Brands, manufacturers, model identifiers and part numbers retain exact source values.

Search, comparison and chat cards share this layer. Comparison rows add `label_zh` and each cell adds `display_value`; both are separate from the raw `dimension`, `value` and its `evidence_id`. Features with source data but no accepted Chinese summary may show the instruction “功能描述请查看英文原文”. Evidence always refers to original facts. No translated title, derived category label or machine-written summary creates an evidence record. Historical USD prices stay in USD and never imply live availability.

`src/choiceagent/localization.py` provides source_hash(product), category_label(raw), field_label(raw), localize_value(field,value), and CatalogLocalizer(path=None).display(product). Product arguments accept Product or mappings. Values are translated by field context: for example Medium means 中号 for Size and 中等笔尖 for Point Type. Category filter values and comparison dimensions must use canonical source identifiers, not their Chinese labels.

The offline builder writes `data/processed/catalog.zh-CN.jsonl` separately from the immutable `catalog.jsonl`. Each sidecar row contains item_id, source_hash, title, category, features, attributes, translation_kind, model, translation_version='zh-CN-v1'. Sidecar attribute keys are original source keys, unlike display attribute labels. The content hash covers exactly title, category, category_path, brand, features[:3], attributes, serialized as UTF-8 JSON with sort_keys=True, ensure_ascii=False, separators=(',',':'). It excludes item_id, price, description and features after the first three. Consumers match both item_id and hash, require the supported translation version, and fall back on a stale or invalid row. Files are published by atomic replacement and reloaded by modification signature; an unavailable sidecar leaves the application usable through controlled fallback.

Offline parsing rejects an untranslated English title and newly introduced numeric values. Each translated feature's numeric values are checked against its corresponding source feature. A truncated JSON output can retain complete, closed title/feature strings but never reconstruct an unfinished string. These checks reduce some generation errors; they do not establish factual equivalence or eliminate the need for human review. Runtime serving does not translate on demand, call an external API, or load a model for this presentation layer. See LOCALIZATION.md and the actual build report for coverage and limitations.

## Retrieval module interface
CatalogRepository(catalog_path:Path, db_path:Path), initialize()->None, get(item_id)->Product|None, all_products()->list[Product], filter_products(constraints)->list[Product], lexical_search(query,limit=200)->list[tuple[str,float]], stats()->dict. Read-only connection per operation, WAL, safe FTS query tokenization.
SearchEngine(repository, index_dir:Path, embedding_model_path:Path|None, device:str='cpu'), search(query,constraints,limit=5,mode='hybrid',liked_item_ids=(),disliked_item_ids=(),personalize=True)->dict. Returns items as serializable cards, applied_constraints, total_eligible,status,warnings,timing_ms. Does not mutate session. close()->None optional.
Products and Constraints are defined in src/choiceagent/schemas.py. No shared DB session tables inside catalog repository.
Semantic encoder default local /datas/wuxi/Projects/datas/models/Qwen3-Embedding-4B (already exists), GPU5; offline build and online query; deterministic normalized embeddings. Include exact FAISS and IDs/model/content manifest. Generation model is exclusively /datas/wuxi/Projects/datas/models/Qwen3-1.7B.

## Local generator (:8001)
Separate server module `python -m choiceagent.model_server`, only one visible GPU default5, Transformers local_files_only=True, thinking disabled, max context8192, max output768, no downloads. POST /v1/chat/completions compatible minimal {messages,temperature,max_tokens,request_id?}, JSON response choices[0].message.content, usage, model, elapsed_ms. Serialize GPU inference, bounded queue/timeouts, validation, readiness health, graceful shutdown. HTTP client provider `LocalModelClient(base_url,timeout)` generate(messages,*,max_tokens,temperature,deadline)->GenerationResult dataclass/text+usage+model. Root Agent handles structured action JSON prompts. No automatic API fallback.

## UI
Streamlit `src/choiceagent/ui.py`, requests HTTP API via httpx, API URL env CHOICEAGENT_API_URL default http://127.0.0.1:8000. Chinese, polished, quick query chips, chat+search tabs, budget/filter sidebar, preference feedback, compare last2, result evidence expander, trace advanced optional, historical USD explicit. Secrets never shown. Graceful backend-down state. No model imports/GPU allocations in UI.
