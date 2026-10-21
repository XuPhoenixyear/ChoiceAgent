# Architecture review: one project bridging retrieval/recommendation and applied agents

Research/design date: 2026-10-20. This is a design proposal, not an implementation or measured result. User: NEU CS academic master's student, June 2028 graduation, first internship, SAILRec accepted at EMNLP 2026, PACERec draft intended for WWW 2027 but not submitted. The project should demonstrate the engineering missing from those papers while retaining their personalized representation theme.

## 1. Cohesive product and scope

Recommended product: **PreferencePilot / 偏好领航：可解释的个性化商品发现与比较助手**. The same retrieval-and-ranking backend supports a conventional search/recommendation page and a conversational tool-using assistant. A user can state a need, refine constraints, compare alternatives and explicitly like/dislike/save items. The assistant must retrieve candidate IDs and evidence, rather than generate a catalog from memory.

A compelling demonstration: “想选一件适合我用途的商品，预算不超过 100 美元；先给我三种选择，并解释为什么适合。” Then add “排除品牌 X；不要把我的临时预算记成长期偏好”, compare two returned IDs, and ask “如果去掉个性化，结果有什么变化？” Historical catalog prices must be labeled as historical USD. An unknown technical attribute remains unknown, not inferred as a hard fact. The exact category should follow data profiling, not be fixed to a tiny subset that cannot support CF.

The deliverable is one coherent system, not a general browser agent, marketplace, transaction system, social application, or model pretraining framework. Ads is a clearly separated optional **sponsored-slot allocation simulator**. It must not be represented as real advertising optimization or evidence of CTR improvement.

## 2. What each audience can inspect

| Search/recommendation interviewer | Agent/application interviewer | Shared evidence |
|---|---|---|
| Data split and leakage audit, ItemCF/MF baselines, semantic retrieval, multi-route recall, fusion, cold-user/item behavior, ranking and tail metrics | Structured intent parsing, typed tools, bounded execution, persistent conversation state, explicit memory, evidence-grounded comparison, timeout recovery, traces and task tests | The exact same item IDs, catalog versions, retrieval backend, ranking rationale, feedback stream, experiment report and reproducible demo |

The project is not supposed to prove mastery of every branch of 搜广推. Its primary proof is search plus recommendation plus personalized Agent; advertising relevance transfers through candidate ranking and constraints, with a separate synthetic exercise if time remains.

## 3. Minimal architecture and boundaries

**Default deployment:** one Python/FastAPI service with modular packages; SQLite for catalog/feedback/session data plus FTS5 lexical index; FAISS for dense retrieval; offline scripts and versioned artifacts; a light web page or Streamlit UI. No Kafka, Kubernetes, Redis, distributed feature store, multi-agent cluster or vector database is required for a first single-user/small-concurrency version. These are possible later replacements, not prerequisites.

```
Offline: raw metadata + reviews
          -> schema/data-quality checks -> time split / version manifest
          -> catalog SQLite + FTS5
          -> content encoder -> item vectors -> FAISS index
          -> train-only interactions -> popularity / ItemCF / optional MF
          -> search judgments + Agent scenarios + frozen evaluation sets

Online: UI -> FastAPI -> session + bounded Agent controller
                           | -> search / recommend tool
                           |       -> constraints -> multi-route candidates
                           |       -> query relevance -> fusion / personalization
                           |       -> diversity -> typed ranked items
                           | -> compare tool -> cited attribute evidence
                           | -> memory tool -> explicit preference records
                           -> evidence validation -> answer + cards + trace

Feedback: explicit like/dislike/save -> event log -> immediate session preference
                                           -> opt-in long-term preference
                                           -> later offline batch retraining
```

Module responsibilities:

1. `data/`: download/stream/filter, schema validation, deduplication by parent item ID, statistics, split manifest, catalog and evidence materialization. No model training inside API startup.
2. `retrieval/`: lexical, semantic and collaborative candidate providers with the same result shape. Keep candidate route scores and ranks for diagnostics.
3. `ranking/`: relevance gating, score/rank fusion, preference boost, exclusions and diversity. Pure callable code, independent of the LLM and HTTP handlers.
4. `agent/`: typed plan, bounded state transitions, tool calls and answer grounding. It cannot write arbitrary SQL, execute shell commands or invent item IDs.
5. `memory/`: explicit preferences with scope, provenance and deletion. It cannot silently turn every chat statement into a permanent user trait.
6. `api/` and `ui/`: orchestration, user/session isolation, request IDs, streaming status, result rendering and idempotent feedback.
7. `evaluation/`: three separate suites for recommendation, search and Agent behavior, plus load/latency measurement. Evaluation data must never silently become training material.

Single-process limitations must be stated. SQLite is appropriate for a local portfolio demo with modest write concurrency; throughput measurements are for the stated machine/process settings, not claims of internet-scale architecture.

## 4. Data contracts

Minimum persistent records:

- `catalog_item(item_id=parent_asin, title, category, brand?, historical_price_usd?, attributes_json, metadata_source, catalog_version, metadata_observed_at?, embedding_version)`.
- `interaction(user_id_hash, item_id, rating, review_time, split, source='amazon_review')`. Review existence is a proxy interaction, not an impression/click/purchase log. `verified_purchase` is a field with its documented meaning, not a full transaction record.
- `evidence(evidence_id, item_id, field_path, value, source_type, source_locator, snapshot_version)`. Product facts use metadata; review-derived claims must be labeled as review opinions.
- `preference(preference_id, user_id, key, value, scope=session|persistent, provenance=explicit|inferred, evidence_event_id, confidence?, updated_at, expires_at?)`.
- `feedback(event_id, user_id, session_id, request_id, item_id, action, timestamp, origin=real_demo|synthetic_test)`. Deduplicate `event_id`; reject feedback on an unknown or unserved item if the endpoint promises impression-linked events.
- `run_trace(trace_id, session_id, catalog/model/index_version, input, plan, tools, timings, token_usage, output_ids, error_class)`; do not log secrets or unnecessary raw personal information.

Do not join metadata using variant `asin` when the metadata is keyed by `parent_asin`. Do not use dataset-wide average ratings/review counts as past-time training features without acknowledging that they can include future information. Safer defaults: train-only popularity and train-only aggregated ratings. Catalog text is a static snapshot; unless the original publication time is reliably available, evaluation is under a known-catalog assumption rather than proof that all metadata existed before each historical interaction.

## 5. Concrete API and tool surface

A small public API is sufficient:

| Endpoint | Request | Response |
|---|---|---|
| `POST /v1/search` | query, optional user_id, constraints, k, ablation flag | ranked items, route ranks, evidence IDs, applied constraints, versions, timings |
| `POST /v1/recommend` | user_id or explicit liked item IDs, optional category, k | ranked items, cold-start fallback flag, explanation features |
| `POST /v1/compare` | 2–4 validated item IDs, selected dimensions | structured comparison with evidence IDs and unknown fields |
| `POST /v1/chat` | session_id, message, request_id | response, item cards, citations, trace_id, clarification if necessary |
| `POST /v1/feedback` | event_id, request_id, item_id, like/dislike/save | event receipt, effective session preference version |
| `GET/DELETE /v1/preferences` | user/session scope | inspect/delete stored preferences |
| `GET /v1/traces/{trace_id}` | local demo authorized access | tools, retrieval/ranking stages, cost and failure states |

Agent tools should be thin wrappers around backend functions, not HTTP requests back into the same process:

```
search_catalog(query, constraints, user_context_id, limit)
recommend_items(user_context_id, constraints, limit)
compare_items(item_ids, dimensions)
read_preferences(user_context_id)
set_preference(key, value, scope, explicit_user_evidence)
```

Every tool uses a Pydantic/JSON-schema input and output contract. The answer generator receives only selected result IDs and corresponding evidence, not the entire catalog. Public API and Agent tools call the same retrieval code so that a gain in backend ranking actually benefits the assistant.

Example search response:

```json
{
  "request_id": "r_001",
  "catalog_version": "catalog_v1",
  "profile_version": 3,
  "items": [
    {
      "item_id": "CATALOG_ID",
      "rank": 1,
      "route_ranks": {"lexical": 4, "semantic": 2, "collaborative": 9},
      "evidence_ids": ["e_title_001", "e_price_001"],
      "reason_codes": ["query_match", "similar_to_explicit_like"],
      "missing_attributes": ["battery_life"]
    }
  ],
  "warnings": ["price_is_historical_usd"],
  "timing_ms": {"retrieval": 0, "ranking": 0}
}
```

Zeros above are placeholders in a schema example, not measured latency.

## 6. Retrieval and ranking that does not confuse spaces or targets

Start with three baselines: train-only popularity, ItemCF and content similarity. An optional MF/BPR model is enough for learned collaborative representations; avoid requiring the full SAILRec or PACERec training pipeline to get a runnable version.

For a query request:

1. Parse query and explicit constraints. Only reliably supported catalog attributes can become hard filters. Unknown price is not “within budget”; either exclude under a strict budget or return an explicitly labeled unknown-price group after permission to relax.
2. Retrieve lexical top-200, semantic top-200 and collaborative top-200 (when user history exists), restricted to the plausible category. Exact limits are configuration values to tune, not guaranteed optima.
3. Union and deduplicate. Apply hard constraints and query relevance gating so personalized but off-topic items cannot overwhelm a request for a different item type. When a filter leaves too few candidates, increase retrieval depth once or explain the shortage; do not silently loosen the constraint.
4. Use rank-based fusion first, with deterministic preference boost, then diversity/redundancy control. RRF is a practical baseline because unrelated route scores are not directly calibrated. Any weighted sum of raw scores needs calibration and validation, not arbitrary addition of BM25, cosine and MF logits.
5. Return about 10 structured items; the Agent can present 3–5 with item-specific evidence. Comparisons may fetch deeper attribute evidence for only 2–4 finalists.

For recommendation without a query, skip query relevance but retain category/dislike and known constraints; use history-weighted ItemCF/semantic candidates and popular fallbacks. For a completely new user, request 2–3 explicit likes or interests, or show a nonpersonalized category sample. Do not allocate a random MF embedding and present it as personalization.

CF vectors and semantic vectors generally occupy different spaces. They can be separate candidate routes or joined by a trained adapter with its own ablation. Appending/adding them without calibration is not a sound baseline. New items can use semantic vectors immediately; any learned adapter must be fixed and applicable to unseen items. New users can use recent explicit liked items rather than requiring a user ID learned offline.

Start with FAISS `IndexFlatIP` on normalized semantic vectors to establish the exact retrieval baseline. Only introduce HNSW when the latency/scale experiment justifies it, and report ANN recall against the exact index. A 50,000-item, 512-d float32 matrix is about 102.4 MB before index overhead; this is not a GPU-only retrieval problem. Encoder inference cost, tokenization, cache misses and LLM calls must be measured separately from vector search.

## 7. A bounded Agent that demonstrates actual application engineering

Suggested state machine:

```
RECEIVE -> LOAD_CONTEXT -> PARSE_INTENT
   -> CLARIFY (only if a critical constraint is missing/conflicting)
   -> TOOL_PLAN -> EXECUTE_TOOLS -> VALIDATE_EVIDENCE
   -> COMPOSE -> VERIFY_IDS_AND_FACTS -> RETURN
                      |
                 RECOVER_ONCE / SAFE_PARTIAL_RESULT
```

Per turn: at most two generation rounds by default (intent/tool plan and grounded answer), four read-tool calls, one argument correction/retry, an explicit end-to-end timeout. These are initial budgets to tune and document. The tool plan may choose search versus recommendation, then comparison; it is not a mandatory fixed tool sequence for every query. Important state transitions, tool inputs, item IDs and failures are shown in a trace panel.

Failure behavior is part of the project: invalid item ID is rejected, zero matches leads to a clarification rather than invented products, a vector retrieval error falls back to lexical retrieval, an LLM timeout returns structured item cards, and malformed memory writes cannot silently persist. Retrieved descriptions and reviews are untrusted data; embedded text such as “ignore previous instructions” is content, never authorization or a system message. Test such examples specifically.

Do not advertise free-form autonomous planning if implementation is a bounded shopping workflow. The substantive Agent skills are typed tool use, state management, evidence checking, failure recovery and evaluation, which are valuable without multi-agent orchestration.

## 8. Memory and feedback semantics

Separate three things:

1. **Session constraints:** “this time less than 100 USD”; retained in this conversation and overridden by a newer explicit statement.
2. **Explicit persistent preferences:** “remember that I prefer lightweight products”; stored only when the user asks to remember or explicitly opts in, with inspect/delete controls.
3. **Inferred preference features:** recent likes/history; treated as uncertain ranking evidence, not asserted personal facts. A single disliked product must not become a permanent broad brand/category dislike unless the user said so.

Immediate demo feedback changes session ranking, not the offline model weights. Batch retraining is a versioned optional later action. Keep `real_demo` and `synthetic_test` events separate; never use synthetic traces to claim real product engagement. Cross-session isolation tests matter more than elaborate vector memory infrastructure here.

Priority order for conflicts: current explicit user instruction > current session constraint > persistent explicit preference > inferred preference. If two current constraints conflict, ask the user to resolve them rather than silently dropping one.

## 9. How the papers connect without copying a thesis into the MVP

- SAILRec contributes the insight that collaborative evidence should be both aligned and deliberately used. In this project the practical counterpart is an inspectable preference context and ablations that turn collaborative evidence on/off while preserving query relevance. A production Agent does not need to reproduce attention-logit modifications to demonstrate this connection.
- PACERec contributes semantic–behavioral adaptation and cached item representations. MVP: precompute text embeddings and compare semantic-only, collaborative-only and fused retrieval. Stretch: train a small shared projection/adapter using only train interactions, compare it against equal-parameter MLP/residual and plain fusion, and measure cold-item behavior plus latency. The paper's full flow objective is not a completion prerequisite.
- The new contribution to the user's profile is serving, data contracts, reproducibility, ranking/Agent joint evaluation, latency and failure analysis. A copied paper checkpoint behind a chat interface would not close those gaps.

## 10. Feasible staged deliverables

Assume 20–25 focused hours per week; change pace if this is inaccurate. Offer a first demo after two weeks, not a six-week hidden build.

| Week | Required result | Completion gate |
|---|---|---|
| 1 | Data manifest, profiling, clean catalog, chronological splits, Popularity + ItemCF baselines | Reproducible data counts; no train/test overlap; run a baseline evaluation from a fresh environment |
| 2 | Lexical + semantic retrieval, constraints, simple web search/recommend page | Search and recommend APIs return catalog-valid IDs; query/recommend evaluation separate; first 3-minute demo |
| 3 | Collaborative fusion, cold-start routes, one ranking ablation matrix | Frozen evaluation script and comparison table; item/filter evidence inspectable |
| 4 | Bounded Agent, comparison tool, explicit memory, trace viewer | Multi-turn scenarios including corrections and constraint conflicts pass; no fabricated IDs |
| 5 | Feedback integration, recovery, grounding and latency/cost experiments | Forced timeouts/empty results/malicious catalog text tested; documented hardware and measurement conditions |
| 6 | Reproducibility polish, demo video, README, limitations, truthful resume bullets | One-command setup with a small sample, all evaluation scripts, version manifest, actual results and failure examples |

Optional extensions compete for time: a small semantic–CF adapter OR a sponsored-slot simulator OR a larger-scale ANN experiment. Choose one only after the core is stable. Do not simultaneously add sequence models, LoRA, RL, distributed processing and a multi-agent framework.

## 11. Data/measurement boundaries that the project must state

Amazon Reviews 2023 contains product metadata and review-derived interactions, not native query/click/exposure/conversion logs. Therefore:

- Recommendation holdout performance measures recovery of future review-associated items under the declared split, not purchase intent, CTR or business revenue.
- Search relevance requires a separate query–item judgment set. LLM-generated queries can seed a development set, but evaluation claims must identify human-reviewed judgments and keep generation/test independence.
- Agent task success requires task/constraint/evidence judgments, not merely whether the LLM replied.
- Metadata prices are historical USD values at crawling, not current live prices. Stock, shipping and transaction success are outside the public dataset.
- A hypothetical sponsored catalog can prove auction/ranking constraint implementation. Bids, budgets, clicks and rewards generated by the project are simulations. Do not derive calibrated pCTR from review ratings and do not report simulated uplift as user behavior.

## 12. Primary sources checked

1. Amazon Reviews 2023 dataset documentation: https://amazon-reviews-2023.github.io/ — fields, user/review and metadata distinctions, `parent_asin` join key, historical USD price, time range and category sizes. The above design draws only those schema facts; numeric dataset totals should be cited from the page if repeated in the final design.
2. SQLite FTS5 documentation: https://www.sqlite.org/fts5.html — local full-text search and built-in BM25 function. Query syntax must be escaped/constructed safely.
3. FAISS official index guide: https://github.com/facebookresearch/faiss/wiki/Faiss-indexes — normalized-vector cosine through `IndexFlatIP`, exact vs HNSW tradeoffs and memory accounting. Index behavior depends on selected implementation/version.
4. Qwen3-Embedding-0.6B official model card: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B — a candidate multilingual local encoder. Pin model revision and verify actual hardware performance; choosing it does not imply it is the universal best embedding model.

The above sources support technical capabilities and dataset semantics. Product choices, timelines, budgets and architecture are design judgments rather than external empirical claims.

## 13. Resource update: A6000 server + Windows RTX 3060, local-first inference

User subsequently confirmed one A6000 server and a local Windows RTX 3060, initially zero API budget and possibly a cheap DeepSeek API later. This changes deployment convenience, not the project scope: **local inference is the default and the entire required path must work without paid API calls**. GPU availability does not justify adding full paper reproduction or large-model finetuning to the critical path.

Recommended deployment:

```
Windows browser -> local SSH tunnel -> server 127.0.0.1: application
                                          -> 127.0.0.1: local generation endpoint
                                          -> local catalog / FAISS / model artifacts
```

Keep application and model servers bound to loopback; use the existing SSH access path. No public server exposure is necessary for a portfolio video or live interview demo. The local RTX 3060 is optional for smoke tests/small encoders; its actual VRAM variant is unknown, so do not promise an 8B model fits unquantized. Server memory, storage and installed drivers must be checked at implementation time rather than assumed from the GPU name.

Model choices verified against official cards on 2026-10-20:

- Default generator candidate: `Qwen/Qwen3-4B-Instruct-2507`, non-thinking instruction model, official card https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 . Start with 8K context and bounded output for this application even though the card advertises a much longer maximum. Model-card maximum context is not an instruction to allocate that much KV cache.
- Alternative local generator: `Qwen/Qwen3-8B`, https://huggingface.co/Qwen/Qwen3-8B . Compare only if 4B fails the frozen development tasks; select on measured tool-schema validity, grounded task success and latency, not parameter count alone.
- Small local encoder: `Qwen/Qwen3-Embedding-0.6B`, https://huggingface.co/Qwen/Qwen3-Embedding-0.6B . Cache catalog vectors offline, and pin model revision/dimension/instruction. The query encoder must match the indexed representation version.
- Optional relevance reranker: `Qwen/Qwen3-Reranker-0.6B`, https://huggingface.co/Qwen/Qwen3-Reranker-0.6B . Only score a small candidate shortlist; add this after a reliable BM25+dense baseline. It scores query relevance, not automatically individual preference.

Avoid making three simultaneous GPU-resident model servers a requirement. Phase the work: embed catalog offline, release that model, train small CF/adapter separately, then run the generator for the demo. Query embeddings can use a separately bounded small encoder on CPU or GPU according to measured latency. A6000 co-residency may be practical, but document memory allocations and check it; do not assume multiple vLLM processes can each reserve most VRAM without conflict. Model loading/cold start is measured separately from warm request latency.

Use a provider-neutral application interface:

```python
class GenerationProvider:
    async def generate(self, messages, output_schema, max_tokens, timeout_s): ...
    async def health(self): ...
```

Implement `LocalProvider` first. Add `DeepSeekProvider` only as an optional adapter using the same messages, typed result and trace contract. “OpenAI-compatible” endpoints may still differ in tool/schema behavior, so add capability flags and contract tests. Application code must not depend on a provider-specific client object outside the adapter. Native tool calling is optional: a validated JSON plan plus controlled dispatch can provide the same bounded workflow and is easier to make portable.

Budget behavior for optional remote inference: disabled by default; explicitly configured key and provider; request token caps; maximum per-task and cumulative spend; reserve estimated request cost against the budget before dispatch; reconcile actual usage afterward; stop or fall back to local inference when the cap is reached. Use current official rates at implementation time instead of hard-coding an unverified price in this design. Do not automatically route to a paid endpoint because the local model timed out. This design/research task makes no paid API calls.
