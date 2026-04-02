# Spec & Design: HF Hub Model Family Popularity Index

## 1. Goal

Build a script that queries the Hugging Face Hub API to produce a **popularity index of model families** (as defined by `config.model_type`), ranked by:

- **Model count** — how many repos on the Hub use that architecture
- **Total downloads** (30-day) — sum of `downloads` across all repos in the family
- **Total all-time downloads** — sum of `downloads_all_time`
- **Total likes** — sum of `likes`
- **Mean trending score** — average `trendingScore` across the family
- **Max trending score** — peak trending model per family

The output will be joined with the local architecture map (`artifacts/transformers_arch_map.json`) so each family row also carries modality, architecture type, task heads, and backend coverage from the codebase.

---

## 2. API Surface (Validated)

### 2.1 Python SDK

```
pip install huggingface_hub>=0.25
```

**Key call:** `HfApi.list_models()`

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `filter` | `"transformers"` | Only models using the Transformers library |
| `sort` | `"downloads"` | Server-side sort (also `"trendingScore"`, `"likes"`, `"created"`) |
| `limit` | `int` or `None` | Cap results; `None` = all |
| `expand` | `list[str]` | Extra fields to include per model |

**Useful `expand` values:**

| Field | Returns | Notes |
|-------|---------|-------|
| `config` | `dict` with `model_type`, `architectures`, etc. | **Primary join key** — `model_type` maps to local family |
| `trendingScore` | `float` | Current trending score |
| `likes` | `int` | Total likes |
| `downloadsAllTime` | `int` | Lifetime downloads |
| `transformersInfo` | `TransformersInfo(auto_model, pipeline_tag, processor)` | Which AutoModel class is used |
| `tags` | `list[str]` | Includes arch names, framework tags |
| `pipeline_tag` | `str` | HF pipeline task label |
| `library_name` | `str` | Usually `"transformers"` or `"sentence-transformers"` |

**Cannot combine** `expand` with `full=True` or `fetch_config=True`.

### 2.2 Performance Characteristics (Measured)

| Operation | Time | Notes |
|-----------|------|-------|
| 1,000 models with `expand=[config]` | ~0.3s | Fast |
| 10,000 models with `expand=[config, trendingScore, likes, downloadsAllTime]` | ~3.3s | Fast |
| 100,000+ models (IDs only, no expand) | ~26s per 50k | Slow, paginated |
| Full enumeration with expand (all fields, all models) | **Estimated 5–15 min** | Hundreds of thousands of models |

### 2.3 Join Key

`ModelInfo.config["model_type"]` → matches the `family_name` in our local arch map (with minor normalization: Hub uses hyphens sometimes, local uses underscores, e.g. `xlm-roberta` ↔ `xlm_roberta`).

---

## 3. Design Options

### Option A: Full Index (All Models)

**Approach:** Stream through every `filter=transformers` model on the Hub with `expand=[config, trendingScore, likes, downloadsAllTime]`. Aggregate in memory.

| Pros | Cons |
|------|------|
| Exact counts and totals | Slow (~5–15 min for 200k+ models) |
| Complete coverage | Large memory footprint |
| No sampling bias | Rate-limit sensitive (unauthenticated) |

### Option B: Top-N Stratified Sampling (Recommended)

**Approach:** Fetch the top N models sorted by `downloads` (e.g., N=50,000). This captures the vast majority of aggregate downloads/likes/trending since popularity follows a power law distro. Then compute family-level aggregates from this sample.

| Pros | Cons |
|------|------|
| Fast (~15–20s for 50k) | Undercounts tiny/new families |
| Captures 95%+ of all downloads/likes | Model-count is approximate |
| Low memory | Some niche families may be missing |
| Works well without auth token | — |

**Mitigation for model count:** Do a separate fast pass (no expand, IDs + config only) to count all models per family, then merge with the detailed stats from the top-N.

### Option C: Hybrid (Recommended Final)

1. **Pass 1 — Count pass:** `list_models(filter="transformers", expand=["config"])` — stream ALL models, extract only `config.model_type` to count models per family. Lightweight per-record, ~2–4 min.
2. **Pass 2 — Stats pass:** `list_models(filter="transformers", sort="downloads", limit=50000, expand=["config", "trendingScore", "likes", "downloadsAllTime", "transformersInfo"])` — get detailed stats for the top 50k. ~15s.
3. **Merge:** Join count-pass family counts with stats-pass aggregates.
4. **Join with local arch map:** Normalize `model_type` and merge with `transformers_arch_map.json`.

This gives exact model counts AND accurate aggregate stats for all meaningful families.

---

## 4. Proposed Implementation

### 4.1 Script

```
tools/build_hub_popularity_index.py
```

**Dependencies:** `huggingface_hub` (already installed), Python stdlib only otherwise.

### 4.2 CLI Interface

```bash
# Default: hybrid approach, top 50k for stats
python3 tools/build_hub_popularity_index.py

# Full index (slow, all models with full expand)
python3 tools/build_hub_popularity_index.py --full

# Custom top-N
python3 tools/build_hub_popularity_index.py --top-n 100000

# Skip count pass (faster, approximate counts)
python3 tools/build_hub_popularity_index.py --skip-count-pass

# Use auth token for higher rate limits
python3 tools/build_hub_popularity_index.py --token $HF_TOKEN
```

### 4.3 Output Artifacts

| File | Format | Content |
|------|--------|---------|
| `artifacts/hub_popularity_index.json` | JSON | Full per-family stats + metadata |
| `artifacts/hub_popularity_report.md` | Markdown | Ranked tables, charts-ready data |
| `artifacts/hub_family_popularity.csv` | CSV | One row per family, all metrics |

### 4.4 JSON Schema

```json
{
  "generated_at": "ISO-8601",
  "parameters": {
    "top_n_for_stats": 50000,
    "total_models_scanned_count_pass": 200000,
    "total_models_scanned_stats_pass": 50000,
    "mode": "hybrid"
  },
  "families": [
    {
      "family_name": "llama",
      "hub_model_type": "llama",
      "model_count": 12345,
      "total_downloads_30d": 999999999,
      "total_downloads_all_time": 9999999999,
      "total_likes": 55555,
      "mean_trending_score": 12.5,
      "max_trending_score": 699.0,
      "top_model_by_downloads": "meta-llama/Llama-3.3-70B-Instruct",
      "top_model_by_trending": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
      "most_common_pipeline_tag": "text-generation",
      "most_common_auto_model": "AutoModelForCausalLM",
      "local_arch_match": {
        "matched": true,
        "local_family_name": "llama",
        "modality": "text",
        "architecture_type": "decoder_only",
        "task_heads": ["LlamaForCausalLM", "LlamaForSequenceClassification", "..."],
        "backends": ["PyTorch"]
      }
    }
  ],
  "rankings": {
    "by_model_count": ["llama", "bert", "gpt2", "..."],
    "by_downloads_30d": ["bert", "llama", "..."],
    "by_downloads_all_time": ["bert", "..."],
    "by_likes": ["llama", "..."],
    "by_trending": ["...", "..."]
  },
  "unmatched_hub_types": [
    { "model_type": "some_custom_type", "model_count": 5, "note": "No local family match" }
  ]
}
```

### 4.5 Markdown Report Sections

1. **Executive Summary** — total models scanned, unique families, top 10 by each metric
2. **Top 50 Families by Model Count** — table
3. **Top 50 Families by Downloads (30d)** — table
4. **Top 50 Families by Likes** — table
5. **Top 50 Families by Trending** — table
6. **Combined Ranking** — composite score (weighted rank across metrics)
7. **LLM Focus** — decoder-only families ranked by downloads
8. **Unmatched Types** — Hub model types with no local arch map match
9. **Coverage** — what % of Hub models have a local family match

---

## 5. Key Design Decisions

### 5.1 `model_type` Normalization

The Hub `config.model_type` and local `family_name` use slightly different conventions:

| Hub `model_type` | Local folder | Normalization |
|-------------------|--------------|---------------|
| `xlm-roberta` | `xlm_roberta` | Replace `-` with `_` |
| `openai-gpt` | `openai` | Lookup table for known aliases |
| `gpt_bigcode` | `gpt_bigcode` | Direct match |

**Approach:** Build a normalization map from the local arch map's config files. Each `configuration_*.py` defines `model_type = "..."` which is the exact string used on the Hub. Parse these to build a definitive `model_type → family_name` mapping.

### 5.2 Composite Popularity Score

To produce a single "popularity" ranking, use a weighted normalized rank:

```
composite_score = (
    0.30 * norm_rank(model_count) +
    0.30 * norm_rank(downloads_30d) +
    0.15 * norm_rank(likes) +
    0.15 * norm_rank(trending) +
    0.10 * norm_rank(downloads_all_time)
)
```

Where `norm_rank(x) = 1 - (rank / max_rank)` so higher is better.

### 5.3 Rate Limits

- Unauthenticated: ~100 req/min, pages of 100 models each → ~10k models/min
- Authenticated (`$HF_TOKEN`): much higher limits
- The script should respect rate limits with exponential backoff (handled by `huggingface_hub` internally)

### 5.4 Caching

- Optionally cache the count-pass results to `artifacts/.hub_count_cache.json` so re-runs can skip the slow pass
- Cache invalidation: timestamp-based, default 24h TTL

---

## 6. Implementation Plan

| Step | Description | Effort |
|------|-------------|--------|
| 1 | Parse `model_type` from all local `configuration_*.py` files to build normalisation map | Small |
| 2 | Implement count pass (stream all models, extract `model_type`, count) | Small |
| 3 | Implement stats pass (top-N with full expand, aggregate per family) | Medium |
| 4 | Merge count + stats, join with local arch map | Small |
| 5 | Compute rankings and composite score | Small |
| 6 | Emit JSON, Markdown, CSV | Medium |
| 7 | Add CLI args (`--full`, `--top-n`, `--skip-count-pass`, `--token`) | Small |
| 8 | Test run, validate output | Small |

---

## 7. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Hub API changes or downtime | Pin `huggingface_hub` version; graceful error handling |
| Very large model count slows count pass | `--skip-count-pass` flag; caching |
| Some models have no `config.model_type` | Count as `"unknown"`; report separately |
| `model_type` doesn't map cleanly to local families | Build explicit alias table from `configuration_*.py` |
| Rate limiting for unauthenticated users | Built-in backoff in `huggingface_hub`; recommend `--token` |

---

## 8. Open Questions for Review

1. **Top-N size:** Is 50,000 sufficient for the stats pass, or should we default higher (e.g., 100k)?
2. **Composite weights:** Are the proposed weights (0.30 count, 0.30 downloads, 0.15 likes, 0.15 trending, 0.10 all-time) reasonable?
3. **Authentication:** Should the script require a HF token, or default to unauthenticated?
4. **Incremental updates:** Should we support delta updates (only fetch new models since last run), or always full refresh?
5. **Scope:** Should we also index non-transformers models (e.g., `diffusers`, `sentence-transformers` as separate library filters)?
