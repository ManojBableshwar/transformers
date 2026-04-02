#!/usr/bin/env python3
"""
Fetch HF Hub popularity data for the top 50k transformers models and merge
with the local architecture map to produce an enriched report.

Outputs:
  - artifacts/transformers_arch_map.json   (enriched with hub_stats)
  - artifacts/transformers_arch_report.md  (family table sorted by downloads)
  - artifacts/transformers_family_summary.csv
  - artifacts/transformers_task_matrix.csv
  - artifacts/hub_popularity_index.json    (raw hub aggregates)

Usage:
    python3 tools/build_hub_popularity_index.py
"""

import ast
import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    from huggingface_hub import HfApi
except ImportError:
    print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub", file=sys.stderr)
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_PATH = REPO_ROOT / "src" / "transformers" / "models"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
ARCH_MAP_PATH = ARTIFACTS_DIR / "transformers_arch_map.json"

TOP_N = 50_000

# ---------------------------------------------------------------------------
# Step 1: Build model_type -> family_name mapping from local config files
# ---------------------------------------------------------------------------

def build_model_type_map() -> dict[str, str]:
    """Parse all configuration_*.py to extract model_type = "..." and map to family folder."""
    model_type_to_family = {}

    for family_dir in sorted(MODELS_PATH.iterdir()):
        if not family_dir.is_dir() or family_dir.name.startswith("_"):
            continue
        family_name = family_dir.name

        for config_file in family_dir.glob("configuration_*.py"):
            try:
                source = config_file.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(source, filename=str(config_file))
            except (SyntaxError, ValueError):
                continue

            for node in ast.walk(tree):
                # Look for class-level: model_type = "something"
                if isinstance(node, ast.ClassDef):
                    for item in node.body:
                        if (isinstance(item, ast.Assign)
                                and len(item.targets) == 1
                                and isinstance(item.targets[0], ast.Name)
                                and item.targets[0].id == "model_type"
                                and isinstance(item.value, ast.Constant)
                                and isinstance(item.value.value, str)):
                            mt = item.value.value
                            # Skip sub-config types (text_model, vision_model, etc.)
                            # Only keep the primary model_type for the family
                            if mt not in model_type_to_family:
                                model_type_to_family[mt] = family_name

    # Also add normalized variants (hyphen <-> underscore)
    extras = {}
    for mt, fam in list(model_type_to_family.items()):
        alt = mt.replace("-", "_")
        if alt not in model_type_to_family:
            extras[alt] = fam
        alt2 = mt.replace("_", "-")
        if alt2 not in model_type_to_family:
            extras[alt2] = fam
    model_type_to_family.update(extras)

    return model_type_to_family


# ---------------------------------------------------------------------------
# Step 2: Fetch Hub data
# ---------------------------------------------------------------------------

def fetch_hub_models(top_n: int) -> list[dict]:
    """Fetch top-N transformers models from HF Hub sorted by downloads."""
    api = HfApi()
    print(f"Fetching top {top_n:,} transformers models from HF Hub (sorted by downloads)...")
    t0 = time.time()

    models_raw = list(api.list_models(
        filter="transformers",
        sort="downloads",
        limit=top_n,
        expand=[
            "config",
            "trendingScore",
            "likes",
            "downloadsAllTime",
            "transformersInfo",
            "tags",
            "pipeline_tag",
            "library_name",
        ],
    ))
    elapsed = time.time() - t0
    print(f"  Fetched {len(models_raw):,} models in {elapsed:.1f}s")

    records = []
    for m in models_raw:
        model_type = None
        if m.config and isinstance(m.config, dict):
            model_type = m.config.get("model_type")

        author = None
        if m.id and "/" in m.id:
            author = m.id.split("/")[0]

        auto_model = None
        pipeline_tag_ti = None
        if m.transformers_info:
            auto_model = getattr(m.transformers_info, "auto_model", None)
            pipeline_tag_ti = getattr(m.transformers_info, "pipeline_tag", None)

        records.append({
            "id": m.id,
            "author": author,
            "model_type": model_type,
            "downloads_30d": m.downloads or 0,
            "downloads_all_time": m.downloads_all_time or 0,
            "likes": m.likes or 0,
            "trending_score": m.trending_score or 0.0,
            "pipeline_tag": m.pipeline_tag or pipeline_tag_ti or "",
            "auto_model": auto_model or "",
            "tags": m.tags or [],
        })

    return records


# ---------------------------------------------------------------------------
# Step 3: Aggregate per family
# ---------------------------------------------------------------------------

def aggregate_by_family(records: list[dict], type_map: dict[str, str]) -> dict[str, dict]:
    """Aggregate Hub records into per-family stats."""
    family_stats = defaultdict(lambda: {
        "model_count": 0,
        "authors": set(),
        "downloads_30d": 0,
        "downloads_all_time": 0,
        "likes": 0,
        "trending_scores": [],
        "top_model_by_downloads": ("", 0),
        "top_model_by_trending": ("", 0.0),
        "pipeline_tags": Counter(),
        "auto_models": Counter(),
        "hub_model_type": None,
    })

    unmatched = defaultdict(lambda: {"count": 0, "downloads": 0})

    for r in records:
        mt = r["model_type"]
        if mt is None:
            mt = "unknown"

        # Resolve family name
        family = type_map.get(mt)
        if family is None:
            # Try normalized
            family = type_map.get(mt.replace("-", "_"))
        if family is None:
            family = type_map.get(mt.replace("_", "-"))
        if family is None:
            # Record as unmatched
            unmatched[mt]["count"] += 1
            unmatched[mt]["downloads"] += r["downloads_30d"]
            continue

        s = family_stats[family]
        s["model_count"] += 1
        if r["author"]:
            s["authors"].add(r["author"])
        s["downloads_30d"] += r["downloads_30d"]
        s["downloads_all_time"] += r["downloads_all_time"]
        s["likes"] += r["likes"]
        s["trending_scores"].append(r["trending_score"])
        if s["hub_model_type"] is None:
            s["hub_model_type"] = mt

        if r["downloads_30d"] > s["top_model_by_downloads"][1]:
            s["top_model_by_downloads"] = (r["id"], r["downloads_30d"])
        if r["trending_score"] > s["top_model_by_trending"][1]:
            s["top_model_by_trending"] = (r["id"], r["trending_score"])

        if r["pipeline_tag"]:
            s["pipeline_tags"][r["pipeline_tag"]] += 1
        if r["auto_model"]:
            s["auto_models"][r["auto_model"]] += 1

    # Finalize
    result = {}
    for family, s in family_stats.items():
        trending = s["trending_scores"]
        result[family] = {
            "model_count": s["model_count"],
            "unique_authors": len(s["authors"]),
            "downloads_30d": s["downloads_30d"],
            "downloads_all_time": s["downloads_all_time"],
            "likes": s["likes"],
            "mean_trending": round(sum(trending) / len(trending), 2) if trending else 0,
            "max_trending": max(trending) if trending else 0,
            "top_model_by_downloads": s["top_model_by_downloads"][0],
            "top_model_by_trending": s["top_model_by_trending"][0],
            "most_common_pipeline_tag": s["pipeline_tags"].most_common(1)[0][0] if s["pipeline_tags"] else "",
            "most_common_auto_model": s["auto_models"].most_common(1)[0][0] if s["auto_models"] else "",
            "hub_model_type": s["hub_model_type"],
        }

    unmatched_list = [
        {"model_type": mt, "count": d["count"], "downloads_30d": d["downloads"]}
        for mt, d in sorted(unmatched.items(), key=lambda x: -x[1]["downloads"])
    ]

    return result, unmatched_list


# ---------------------------------------------------------------------------
# Step 4: Merge with local arch map and regenerate report
# ---------------------------------------------------------------------------

def load_arch_map() -> dict:
    """Load existing architecture map JSON."""
    if not ARCH_MAP_PATH.exists():
        print(f"ERROR: {ARCH_MAP_PATH} not found. Run build_transformers_arch_map.py first.", file=sys.stderr)
        sys.exit(1)
    return json.loads(ARCH_MAP_PATH.read_text(encoding="utf-8"))


def merge_and_save(arch_map: dict, hub_stats: dict[str, dict], unmatched: list[dict]):
    """Merge Hub stats into arch map families and save enriched JSON."""
    for fam in arch_map["families"]:
        fname = fam["family_name"]
        if fname in hub_stats:
            fam["hub_stats"] = hub_stats[fname]
        else:
            fam["hub_stats"] = {
                "model_count": 0,
                "unique_authors": 0,
                "downloads_30d": 0,
                "downloads_all_time": 0,
                "likes": 0,
                "mean_trending": 0,
                "max_trending": 0,
                "top_model_by_downloads": "",
                "top_model_by_trending": "",
                "most_common_pipeline_tag": "",
                "most_common_auto_model": "",
                "hub_model_type": "",
            }

    arch_map["hub_metadata"] = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "top_n": TOP_N,
        "families_matched": sum(1 for f in arch_map["families"] if f["hub_stats"]["model_count"] > 0),
        "families_unmatched_on_hub": len([f for f in arch_map["families"] if f["hub_stats"]["model_count"] == 0]),
        "hub_types_unmatched_locally": unmatched,
    }

    # Write enriched arch map
    ARCH_MAP_PATH.write_text(json.dumps(arch_map, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote enriched {ARCH_MAP_PATH}")

    # Write separate hub popularity index
    hub_index_path = ARTIFACTS_DIR / "hub_popularity_index.json"
    hub_index = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "top_n": TOP_N,
        "family_stats": hub_stats,
        "unmatched_hub_types": unmatched,
    }
    hub_index_path.write_text(json.dumps(hub_index, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {hub_index_path}")

    return arch_map


# ---------------------------------------------------------------------------
# Step 5: Generate enriched report
# ---------------------------------------------------------------------------

SUFFIX_TO_TASK = {
    "ForCausalLM": "text-generation",
    "ForMaskedLM": "fill-mask",
    "ForSequenceClassification": "text-classification",
    "ForTokenClassification": "token-classification",
    "ForQuestionAnswering": "question-answering",
    "ForMultipleChoice": "multiple-choice",
    "ForSeq2SeqLM": "text2text-generation",
    "ForConditionalGeneration": "conditional-generation",
    "ForVision2Seq": "image-to-text",
    "ForSpeechSeq2Seq": "automatic-speech-recognition",
    "ForCTC": "ctc-speech-recognition",
    "ForAudioClassification": "audio-classification",
    "ForImageClassification": "image-classification",
    "ForObjectDetection": "object-detection",
    "ForPreTraining": "pretraining",
}


def fmt_num(n: int) -> str:
    """Format large numbers with K/M suffixes."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def generate_report(arch_map: dict):
    """Generate the enriched Markdown report."""
    summary = arch_map["summary"]
    families = arch_map["families"]
    taxonomy = arch_map["task_taxonomy"]
    cross = arch_map["cross_cutting_insights"]
    hub_meta = arch_map.get("hub_metadata", {})

    lines = []
    w = lines.append

    w("# Transformers Architecture Map Report")
    w("")
    w(f"**Generated:** {arch_map['generated_at']}")
    w(f"**Hub data fetched:** {hub_meta.get('fetched_at', 'N/A')}")
    w(f"**Repo root:** `{arch_map['repo_root']}`")
    w("")

    # 1. Executive summary
    w("## 1. Executive Summary")
    w("")
    w("| Metric | Count |")
    w("|--------|-------|")
    w(f"| Model family folders | {summary['family_count']} |")
    w(f"| Total classes parsed | {summary['class_count']} |")
    w(f"| Base model classes | {summary['base_model_class_count']} |")
    w(f"| Task head classes | {summary['task_head_class_count']} |")
    w(f"| Families with Hub models | {hub_meta.get('families_matched', 'N/A')} |")
    w(f"| Families with no Hub models | {hub_meta.get('families_unmatched_on_hub', 'N/A')} |")
    w("")

    w("### Modality Distribution")
    w("")
    w("| Modality | Families |")
    w("|----------|----------|")
    for mod, cnt in sorted(summary["families_by_modality"].items(), key=lambda x: -x[1]):
        w(f"| {mod} | {cnt} |")
    w("")

    w("### Architecture Type Distribution")
    w("")
    w("| Type | Families |")
    w("|------|----------|")
    for atype, cnt in sorted(summary["families_by_architecture_type"].items(), key=lambda x: -x[1]):
        w(f"| {atype} | {cnt} |")
    w("")

    # 2. Family inventory — sorted by downloads
    w("## 2. Family Inventory (Sorted by Downloads)")
    w("")
    w(f"Showing all {len(families)} families. Sorted by 30-day Hub downloads (descending).")
    w("")
    w("| # | Family | Modality | Architecture | Backends | Models | Authors | Downloads | Likes | Base Models | Task Heads | Tasks | Notes |")
    w("|---|--------|----------|--------------|----------|--------|---------|-----------|-------|-------------|------------|-------|-------|")

    sorted_families = sorted(families, key=lambda f: f.get("hub_stats", {}).get("downloads_30d", 0), reverse=True)

    for rank, fam in enumerate(sorted_families, 1):
        hs = fam.get("hub_stats", {})
        backends = []
        if fam["status_flags"]["has_pytorch"]:
            backends.append("PT")
        if fam["status_flags"]["has_tensorflow"]:
            backends.append("TF")
        if fam["status_flags"]["has_flax"]:
            backends.append("Flax")
        backends_str = ", ".join(backends) if backends else "—"

        base_str = ", ".join(fam["base_model_classes"][:3])
        if len(fam["base_model_classes"]) > 3:
            base_str += f" (+{len(fam['base_model_classes']) - 3})"
        if not base_str:
            base_str = "—"

        heads = fam["task_head_classes"]
        if heads:
            heads_str = ", ".join(heads[:3])
            if len(heads) > 3:
                heads_str += f" (+{len(heads) - 3})"
        else:
            heads_str = "—"

        tasks_str = ", ".join(fam["tasks_supported"]) if fam["tasks_supported"] else "—"
        notes_str = "; ".join(fam["notes"]) if fam["notes"] else ""

        mc = hs.get("model_count", 0)
        ac = hs.get("unique_authors", 0)
        dl = hs.get("downloads_30d", 0)
        lk = hs.get("likes", 0)

        mc_str = fmt_num(mc) if mc else "—"
        ac_str = fmt_num(ac) if ac else "—"
        dl_str = fmt_num(dl) if dl else "—"
        lk_str = fmt_num(lk) if lk else "—"

        w(f"| {rank} | {fam['family_name']} | {fam['modality']} | {fam['architecture_type']} | {backends_str} | {mc_str} | {ac_str} | {dl_str} | {lk_str} | {base_str} | {heads_str} | {tasks_str} | {notes_str} |")
    w("")

    # 3. Aggregated task analysis
    w("## 3. Aggregated Task Analysis")
    w("")

    w("### Most Common Task Head Suffixes")
    w("")
    w("| Suffix | Count | Task |")
    w("|--------|-------|------|")
    for item in summary["top_task_suffixes"][:20]:
        task = SUFFIX_TO_TASK.get(item["suffix"], item["suffix"])
        w(f"| {item['suffix']} | {item['count']} | {task} |")
    w("")

    w("### Families per Task")
    w("")
    for task_name in sorted(taxonomy["task_to_families"].keys()):
        fams = taxonomy["task_to_families"][task_name]
        w(f"- **{task_name}** ({len(fams)} families): {', '.join(fams[:10])}" + (" ..." if len(fams) > 10 else ""))
    w("")

    w("### Top Families by Head Count")
    w("")
    w("| Family | Head Count |")
    w("|--------|------------|")
    for item in summary["top_families_by_head_count"]:
        w(f"| {item['family']} | {item['head_count']} |")
    w("")

    # 4. LLM-relevant view
    w("## 4. LLM-Relevant View")
    w("")

    w("### Families with ForCausalLM (text-generation) — by Downloads")
    w("")
    causal_fams = set(cross["families_with_causal_lm"])
    causal_sorted = sorted(
        [f for f in families if f["family_name"] in causal_fams],
        key=lambda f: f.get("hub_stats", {}).get("downloads_30d", 0),
        reverse=True,
    )
    w("| Family | Downloads | Models | Likes |")
    w("|--------|-----------|--------|-------|")
    for f in causal_sorted[:40]:
        hs = f.get("hub_stats", {})
        w(f"| {f['family_name']} | {fmt_num(hs.get('downloads_30d', 0))} | {fmt_num(hs.get('model_count', 0))} | {fmt_num(hs.get('likes', 0))} |")
    w("")

    w("### Families with ForConditionalGeneration")
    w("")
    cond_fams = set(cross["families_with_conditional_generation"])
    cond_sorted = sorted(
        [f for f in families if f["family_name"] in cond_fams],
        key=lambda f: f.get("hub_stats", {}).get("downloads_30d", 0),
        reverse=True,
    )
    w("| Family | Downloads | Models | Likes |")
    w("|--------|-----------|--------|-------|")
    for f in cond_sorted[:30]:
        hs = f.get("hub_stats", {})
        w(f"| {f['family_name']} | {fmt_num(hs.get('downloads_30d', 0))} | {fmt_num(hs.get('model_count', 0))} | {fmt_num(hs.get('likes', 0))} |")
    w("")

    w("### Families with ForVision2Seq (image-to-text)")
    w("")
    v2s_fams = cross["families_with_vision2seq"]
    w(f"**{len(v2s_fams)} families:** {', '.join(sorted(v2s_fams))}")
    w("")

    # 5. Backend coverage
    w("## 5. Backend Coverage")
    w("")
    pytorch_only = []
    pt_tf = []
    pt_flax = []
    all_three = []

    for fam in families:
        sf = fam["status_flags"]
        if sf["has_pytorch"] and sf["has_tensorflow"] and sf["has_flax"]:
            all_three.append(fam["family_name"])
        elif sf["has_pytorch"] and sf["has_tensorflow"]:
            pt_tf.append(fam["family_name"])
        elif sf["has_pytorch"] and sf["has_flax"]:
            pt_flax.append(fam["family_name"])
        elif sf["has_pytorch"]:
            pytorch_only.append(fam["family_name"])

    no_backend = [f["family_name"] for f in families if not any(
        f["status_flags"][k] for k in ("has_pytorch", "has_tensorflow", "has_flax")
    )]

    w("| Backend Config | Count |")
    w("|----------------|-------|")
    w(f"| PyTorch only | {len(pytorch_only)} |")
    w(f"| PyTorch + TensorFlow | {len(pt_tf)} |")
    w(f"| PyTorch + Flax | {len(pt_flax)} |")
    w(f"| All three (PT+TF+Flax) | {len(all_three)} |")
    w(f"| No modeling files | {len(no_backend)} |")
    w("")

    if all_three:
        w(f"**All three backends:** {', '.join(sorted(all_three))}")
        w("")

    # 6. Interesting outliers
    w("## 6. Interesting Outliers")
    w("")

    w("### Families with Many Heads (>6)")
    w("")
    for fam in sorted(families, key=lambda f: len(f["task_head_classes"]), reverse=True):
        if len(fam["task_head_classes"]) > 6:
            w(f"- **{fam['family_name']}**: {len(fam['task_head_classes'])} heads")
    w("")

    w("### Shared/Meta Folders")
    w("")
    for fam in families:
        if fam["status_flags"]["is_likely_shared_or_meta"]:
            w(f"- **{fam['family_name']}**: {'; '.join(fam['notes']) if fam['notes'] else 'shared infrastructure'}")
    w("")

    w("### Modular-File Families")
    w("")
    modular_fams = [f["family_name"] for f in families if f["status_flags"]["has_modular_file"]]
    w(f"**{len(modular_fams)} families** use modular files: {', '.join(sorted(modular_fams)[:30])}" + (" ..." if len(modular_fams) > 30 else ""))
    w("")

    w("### Unmatched Hub Model Types")
    w("")
    unmatched = hub_meta.get("hub_types_unmatched_locally", [])
    if unmatched:
        w("Hub `model_type` values with no matching local family folder:")
        w("")
        w("| model_type | Hub Models | Downloads (30d) |")
        w("|------------|-----------|-----------------|")
        for u in unmatched[:30]:
            w(f"| {u['model_type']} | {u['count']} | {fmt_num(u['downloads_30d'])} |")
    else:
        w("All Hub model types matched a local family.")
    w("")

    # 7. Caveats
    w("## 7. Caveats")
    w("")
    w("- Folder count ≠ class count: one family can contain many classes and task heads.")
    w("- Modality and architecture type are heuristic-based and may misclassify specialized families.")
    w("- Some folders are infrastructure (e.g., `auto`), shared utilities, or deprecated wrappers.")
    w("- Hub stats are from the top 50K models by downloads; long-tail models below this threshold are excluded.")
    w("- Model count and author count reflect the top-50K sample, not the full Hub.")
    w("- `model_type` normalization may miss non-standard or custom model types.")
    w("")

    md_path = ARTIFACTS_DIR / "transformers_arch_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {md_path}")


def generate_csv(families: list[dict]):
    """Generate enriched family summary CSV."""
    csv_path = ARTIFACTS_DIR / "transformers_family_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "family_name", "modality", "architecture_type",
            "has_pytorch", "has_tensorflow", "has_flax",
            "base_model_count", "task_head_count",
            "tasks_supported", "has_modular", "is_shared_meta", "is_legacy",
            "hub_model_count", "hub_unique_authors",
            "hub_downloads_30d", "hub_downloads_all_time",
            "hub_likes", "hub_mean_trending", "hub_max_trending",
            "hub_top_model",
        ])
        for fam in sorted(families, key=lambda x: x.get("hub_stats", {}).get("downloads_30d", 0), reverse=True):
            hs = fam.get("hub_stats", {})
            writer.writerow([
                fam["family_name"],
                fam["modality"],
                fam["architecture_type"],
                fam["status_flags"]["has_pytorch"],
                fam["status_flags"]["has_tensorflow"],
                fam["status_flags"]["has_flax"],
                len(fam["base_model_classes"]),
                len(fam["task_head_classes"]),
                "; ".join(fam["tasks_supported"]),
                fam["status_flags"]["has_modular_file"],
                fam["status_flags"]["is_likely_shared_or_meta"],
                fam["status_flags"]["is_likely_legacy"],
                hs.get("model_count", 0),
                hs.get("unique_authors", 0),
                hs.get("downloads_30d", 0),
                hs.get("downloads_all_time", 0),
                hs.get("likes", 0),
                hs.get("mean_trending", 0),
                hs.get("max_trending", 0),
                hs.get("top_model_by_downloads", ""),
            ])
    print(f"Wrote {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Build model_type map
    print("Building model_type → family mapping from local configs...")
    type_map = build_model_type_map()
    print(f"  Mapped {len(type_map)} model_type values to local families")

    # Step 2: Fetch Hub data
    records = fetch_hub_models(TOP_N)

    # Step 3: Aggregate
    print("Aggregating per family...")
    hub_stats, unmatched = aggregate_by_family(records, type_map)
    print(f"  Matched {len(hub_stats)} families, {len(unmatched)} unmatched hub types")

    # Step 4: Load arch map and merge
    print("Loading local architecture map...")
    arch_map = load_arch_map()
    arch_map = merge_and_save(arch_map, hub_stats, unmatched)

    # Step 5: Generate report and CSVs
    generate_report(arch_map)
    generate_csv(arch_map["families"])

    # Console summary
    top_by_dl = sorted(hub_stats.items(), key=lambda x: -x[1]["downloads_30d"])
    print("\n" + "=" * 70)
    print("HUB POPULARITY SUMMARY")
    print("=" * 70)
    print(f"  Total models fetched:    {len(records):,}")
    print(f"  Families matched:        {len(hub_stats)}")
    print(f"  Unmatched hub types:     {len(unmatched)}")
    print()
    print("  Top 15 families by 30-day downloads:")
    print(f"    {'Family':<30s} {'Downloads':>12s} {'Models':>8s} {'Authors':>8s} {'Likes':>8s}")
    for name, s in top_by_dl[:15]:
        print(f"    {name:<30s} {fmt_num(s['downloads_30d']):>12s} {s['model_count']:>8,} {s['unique_authors']:>8,} {s['likes']:>8,}")
    print()
    print("  Top unmatched hub types:")
    for u in unmatched[:10]:
        print(f"    {u['model_type']:<30s} {u['count']:>6} models  {fmt_num(u['downloads_30d']):>10s} downloads")
    print("=" * 70)


if __name__ == "__main__":
    main()
