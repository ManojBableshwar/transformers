#!/usr/bin/env python3
"""
Build a comprehensive architecture map of Hugging Face Transformers model families.

Scans src/transformers/models/ to produce:
  - artifacts/transformers_arch_map.json
  - artifacts/transformers_arch_report.md
  - artifacts/transformers_family_summary.csv
  - artifacts/transformers_task_matrix.csv

Usage:
    python tools/build_transformers_arch_map.py
"""

import ast
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_PATH = REPO_ROOT / "src" / "transformers" / "models"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

# Task suffix -> normalised task name
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
    "ForAudioFrameClassification": "audio-frame-classification",
    "ForAudioXVector": "audio-xvector",
    "ForImageClassification": "image-classification",
    "ForImageSegmentation": "image-segmentation",
    "ForSemanticSegmentation": "semantic-segmentation",
    "ForInstanceSegmentation": "instance-segmentation",
    "ForPanopticSegmentation": "panoptic-segmentation",
    "ForObjectDetection": "object-detection",
    "ForZeroShotObjectDetection": "zero-shot-object-detection",
    "ForZeroShotImageClassification": "zero-shot-image-classification",
    "ForVisualQuestionAnswering": "visual-question-answering",
    "ForDocumentQuestionAnswering": "document-question-answering",
    "ForMaskedImageModeling": "masked-image-modeling",
    "ForImageToImage": "image-to-image",
    "ForDepthEstimation": "depth-estimation",
    "ForVideoClassification": "video-classification",
    "ForPreTraining": "pretraining",
    "ForNextSentencePrediction": "next-sentence-prediction",
    "ForMaskedGeneration": "masked-generation",
    "ForUniversalSegmentation": "universal-segmentation",
    "ForTextToWaveform": "text-to-waveform",
    "ForTextEncoding": "text-encoding",
    "ForImageTextRetrieval": "image-text-retrieval",
    "ForReferringExpressionSegmentation": "referring-expression-segmentation",
    "ForOpenVocabularyDetection": "open-vocabulary-detection",
}

# Regex pattern to detect task-head suffixes (generic fallback)
TASK_SUFFIX_RE = re.compile(r"For[A-Z]\w+$")

# Heuristics for modality classification keywords
VISION_KEYWORDS = {
    "image", "vit", "deit", "beit", "swin", "convnext", "resnet", "poolformer",
    "segformer", "detr", "yolo", "sam", "dinov2", "dinat", "nat", "pvt",
    "efficientnet", "mobilenet", "mobilevit", "regnet", "mask2former",
    "siglip", "eva", "zoedepth", "depth_anything", "bit", "focalnet",
}
AUDIO_KEYWORDS = {
    "wav2vec", "hubert", "whisper", "speech", "audio", "bark", "vits",
    "encodec", "wavlm", "seamless", "musicgen", "unispeech", "mctct",
    "sew", "clap", "spectrogram", "mms", "mimi", "xcodec", "voxtral",
}
MULTIMODAL_KEYWORDS = {
    "clip", "blip", "flamingo", "llava", "paligemma", "idefics", "fuyu",
    "align", "altclip", "bridgetower", "git", "instructblip", "kosmos",
    "vipllava", "mgp_str", "pix2struct", "tvp", "x_clip", "chameleon",
    "aria", "emu", "colpali", "internvl", "molmo", "mllama", "pixtral",
    "qwen2_vl", "qwen2_5_vl", "gemma3",
}
DIFFUSION_KEYWORDS = {"unet", "vqmodel", "vae"}

# Heuristics for shared/meta/infra folders
SHARED_META_NAMES = {
    "auto", "deprecated", "encoder_decoder",
}

LEGACY_INDICATORS = {"deprecated", "legacy"}

# Architecture type heuristic keywords
DECODER_ONLY_INDICATORS = {"ForCausalLM"}
ENCODER_DECODER_INDICATORS = {"ForSeq2SeqLM", "ForConditionalGeneration", "ForSpeechSeq2Seq"}
VISION_ENCODER_INDICATORS = {
    "ForImageClassification", "ForObjectDetection", "ForSemanticSegmentation",
    "ForInstanceSegmentation", "ForPanopticSegmentation", "ForDepthEstimation",
    "ForMaskedImageModeling",
}
SPEECH_INDICATORS = {"ForCTC", "ForAudioClassification", "ForSpeechSeq2Seq", "ForAudioFrameClassification"}
MULTIMODAL_INDICATORS = {"ForVision2Seq", "ForVisualQuestionAnswering", "ForImageTextRetrieval"}


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def extract_classes_from_file(filepath: Path) -> list[dict]:
    """Parse a Python file and return class definitions with names and base classes."""
    try:
        source = filepath.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, ValueError, UnicodeDecodeError):
        return []

    classes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            bases = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.append(ast.dump(base))
            classes.append({
                "class_name": node.name,
                "bases": bases,
                "lineno": node.lineno,
            })
    return classes


def classify_class(class_name: str, source_file: str, family_name: str) -> dict:
    """Classify a class by its name suffix and source file context."""
    backend = "unknown"
    fname = os.path.basename(source_file)

    if fname.startswith("modeling_tf_") or fname.startswith("modeling_tf."):
        backend = "tensorflow"
    elif fname.startswith("modeling_flax_") or fname.startswith("modeling_flax."):
        backend = "flax"
    elif fname.startswith("modeling_"):
        backend = "pytorch"
    elif fname.startswith("configuration_"):
        backend = "config"
    elif fname.startswith("tokenization_"):
        backend = "tokenizer"
    elif fname.startswith("processing_") or fname.startswith("image_processing_") or fname.startswith("feature_extraction_") or fname.startswith("video_processing_"):
        backend = "processor"

    # Determine class_kind and task
    class_kind = "other"
    task_inferred = ""
    task_suffix = ""

    if class_name.endswith("Output") or class_name.endswith("Outputs"):
        class_kind = "other"  # dataclass / named tuple outputs, not real heads
    elif class_name.endswith("Config") or class_name.endswith("OnnxConfig"):
        class_kind = "config"
    elif "Tokenizer" in class_name:
        class_kind = "tokenizer"
    elif "ImageProcessor" in class_name or "FeatureExtractor" in class_name:
        class_kind = "processor"
    elif class_name.endswith("Processor"):
        class_kind = "processor"
    elif class_name.endswith("Model") or class_name.endswith("Encoder") or class_name.endswith("Decoder"):
        # Check it's not something like ForSomeModel (task head)
        if not TASK_SUFFIX_RE.search(class_name):
            class_kind = "base_model"
    elif class_name.endswith("PreTrainedModel") or class_name.endswith("PretrainedModel"):
        class_kind = "other"  # base infrastructure
    else:
        # Try to match a task-head suffix
        for suffix in sorted(SUFFIX_TO_TASK.keys(), key=len, reverse=True):
            if class_name.endswith(suffix):
                class_kind = "task_head"
                task_suffix = suffix
                task_inferred = SUFFIX_TO_TASK[suffix]
                break
        else:
            # Generic fallback: check if it looks like a task head
            m = TASK_SUFFIX_RE.search(class_name)
            if m:
                task_suffix = m.group(0)
                class_kind = "task_head"
                task_inferred = task_suffix  # keep raw

    # Heuristic: if backend is config/tokenizer/processor, override class_kind
    if backend == "config":
        class_kind = "config"
    elif backend == "tokenizer":
        class_kind = "tokenizer"
    elif backend == "processor":
        class_kind = "processor"

    # Re-check for base_model: should have a modeling file backend
    if class_kind == "base_model" and backend not in ("pytorch", "tensorflow", "flax", "unknown"):
        class_kind = "other"

    # Is this likely an AutoModel-compatible export?
    is_auto = False
    if class_kind in ("base_model", "task_head") and backend in ("pytorch", "tensorflow", "flax"):
        # Typically classes that don't start with underscore and aren't internal
        if not class_name.startswith("_"):
            is_auto = True

    return {
        "class_name": class_name,
        "source_file": source_file,
        "backend": backend if backend not in ("config", "tokenizer", "processor") else "unknown",
        "class_kind": class_kind,
        "task_inferred": task_inferred,
        "task_suffix": task_suffix,
        "is_auto_export_candidate": is_auto,
    }


def infer_modality(family_name: str, class_names: list[str], file_names: list[str]) -> str:
    """Infer family modality from naming patterns."""
    fn_lower = family_name.lower()
    all_names = " ".join(class_names).lower() + " " + " ".join(file_names).lower() + " " + fn_lower

    # Check multimodal first (higher priority)
    if any(kw in fn_lower for kw in MULTIMODAL_KEYWORDS):
        return "multimodal"
    if any("image_processing" in f or "video_processing" in f for f in file_names) and any("tokenization" in f for f in file_names):
        return "multimodal"
    if any(kw in all_names for kw in MULTIMODAL_KEYWORDS):
        return "multimodal"

    # Check audio/speech
    if any(kw in fn_lower for kw in AUDIO_KEYWORDS):
        return "audio"
    if any("feature_extraction" in f for f in file_names) and "audio" in all_names:
        return "audio"
    if any(kw in all_names for kw in {"forcttc", "foraudio", "forspeech", "wav2vec", "whisper", "hubert"}):
        return "audio"

    # Check vision
    if any(kw in fn_lower for kw in VISION_KEYWORDS):
        return "vision"
    if any("image_processing" in f for f in file_names) and not any("tokenization" in f for f in file_names):
        return "vision"
    if any(kw in fn_lower for kw in {"detr", "mask2former", "sam", "yolo", "zoedepth", "depth_anything"}):
        return "vision"

    # Check diffusion
    if any(kw in fn_lower for kw in DIFFUSION_KEYWORDS):
        return "diffusion_like"

    # If has only modeling files and no image/audio indicators, likely text
    if any("tokenization" in f for f in file_names) and not any("image_processing" in f for f in file_names):
        return "text"

    # If has modeling files
    if any("modeling_" in f for f in file_names):
        # Check task heads for clues
        suffixes = set()
        for cn in class_names:
            for suffix in SUFFIX_TO_TASK:
                if cn.endswith(suffix):
                    suffixes.add(suffix)
        if suffixes & VISION_ENCODER_INDICATORS:
            return "vision"
        if suffixes & SPEECH_INDICATORS:
            return "audio"
        if suffixes & MULTIMODAL_INDICATORS:
            return "multimodal"
        if suffixes & DECODER_ONLY_INDICATORS:
            return "text"
        if suffixes & ENCODER_DECODER_INDICATORS:
            return "text"
        # Default for modeling-only
        return "text"

    return "unknown"


def infer_architecture_type(family_name: str, class_names: list[str], task_suffixes: set[str], modality: str) -> str:
    """Infer architecture type from class patterns."""

    if modality == "multimodal":
        return "multimodal"
    if modality == "audio":
        return "speech"
    if modality == "diffusion_like":
        return "other"

    # Check task suffixes
    if task_suffixes & MULTIMODAL_INDICATORS:
        return "multimodal"
    if task_suffixes & SPEECH_INDICATORS:
        return "speech"

    has_causal = "ForCausalLM" in task_suffixes
    has_seq2seq = bool(task_suffixes & {"ForSeq2SeqLM", "ForConditionalGeneration"})
    has_vision_heads = bool(task_suffixes & VISION_ENCODER_INDICATORS)
    has_encoder_classes = any("Encoder" in cn for cn in class_names)
    has_decoder_classes = any("Decoder" in cn for cn in class_names)

    if has_causal and not has_seq2seq and not has_vision_heads:
        return "decoder_only"
    if has_seq2seq:
        return "encoder_decoder"
    if has_vision_heads and modality == "vision":
        return "vision_encoder"
    if has_encoder_classes and has_decoder_classes and not has_causal:
        return "encoder_decoder"

    # Fallback heuristics by modality
    if modality == "vision":
        return "vision_encoder"
    if modality == "text":
        if has_causal:
            return "decoder_only"
        # Check for encoder-only patterns (BERT-like)
        mask_fill = task_suffixes & {"ForMaskedLM", "ForSequenceClassification", "ForTokenClassification", "ForQuestionAnswering"}
        if mask_fill and not has_causal:
            return "encoder_only"
        # If only base model, hard to tell
        return "unknown"

    return "unknown"


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_family(family_dir: Path) -> dict:
    """Analyze a single model family directory."""
    family_name = family_dir.name

    # Categorize files
    files_by_type = {
        "configuration": [],
        "modeling": [],
        "modeling_tf": [],
        "modeling_flax": [],
        "tokenization": [],
        "processing": [],
        "other": [],
    }

    all_files = []
    for f in sorted(family_dir.iterdir()):
        if f.is_file() and f.suffix == ".py" and f.name != "__init__.py" and "__pycache__" not in str(f):
            rel = f.name
            all_files.append(rel)
            if rel.startswith("configuration_"):
                files_by_type["configuration"].append(rel)
            elif rel.startswith("modeling_tf_") or rel.startswith("modeling_tf."):
                files_by_type["modeling_tf"].append(rel)
            elif rel.startswith("modeling_flax_") or rel.startswith("modeling_flax."):
                files_by_type["modeling_flax"].append(rel)
            elif rel.startswith("modeling_"):
                files_by_type["modeling"].append(rel)
            elif rel.startswith("tokenization_"):
                files_by_type["tokenization"].append(rel)
            elif rel.startswith("processing_") or rel.startswith("image_processing_") or rel.startswith("feature_extraction_") or rel.startswith("video_processing_"):
                files_by_type["processing"].append(rel)
            else:
                files_by_type["other"].append(rel)

    # Status flags
    has_modular = any(f.startswith("modular_") for f in all_files)
    status_flags = {
        "has_pytorch": len(files_by_type["modeling"]) > 0,
        "has_tensorflow": len(files_by_type["modeling_tf"]) > 0,
        "has_flax": len(files_by_type["modeling_flax"]) > 0,
        "has_tokenizer": any("tokenization_" in f for f in all_files),
        "has_fast_tokenizer": any("tokenization_" in f and "fast" in f.lower() for f in all_files),
        "has_processor": any(f.startswith("processing_") for f in all_files),
        "has_image_processor": any(f.startswith("image_processing_") for f in all_files),
        "has_feature_extractor": any(f.startswith("feature_extraction_") for f in all_files),
        "has_config": len(files_by_type["configuration"]) > 0,
        "has_modular_file": has_modular,
        "is_likely_shared_or_meta": family_name in SHARED_META_NAMES or not any(
            files_by_type[k] for k in ("modeling", "modeling_tf", "modeling_flax")
        ),
        "is_likely_legacy": any(ind in family_name.lower() for ind in LEGACY_INDICATORS),
    }

    # Parse classes from all relevant Python files
    all_classes_raw = []
    source_files_to_parse = (
        files_by_type["configuration"]
        + files_by_type["modeling"]
        + files_by_type["modeling_tf"]
        + files_by_type["modeling_flax"]
        + files_by_type["tokenization"]
        + files_by_type["processing"]
    )

    for fname in source_files_to_parse:
        filepath = family_dir / fname
        raw_classes = extract_classes_from_file(filepath)
        for rc in raw_classes:
            all_classes_raw.append((rc["class_name"], fname))

    # Classify each class
    classes = []
    for cname, fname in all_classes_raw:
        info = classify_class(cname, fname, family_name)
        classes.append(info)

    # Filter out private/internal helper classes that start with _
    # Keep them in the data but don't count as base/task
    base_model_classes = [c["class_name"] for c in classes if c["class_kind"] == "base_model"]
    task_head_classes = [c["class_name"] for c in classes if c["class_kind"] == "task_head"]
    task_suffixes_set = set(c["task_suffix"] for c in classes if c["task_suffix"])
    tasks_supported = sorted(set(c["task_inferred"] for c in classes if c["task_inferred"]))
    task_head_suffixes = sorted(task_suffixes_set)

    class_names_all = [c["class_name"] for c in classes]

    modality = infer_modality(family_name, class_names_all, all_files)
    arch_type = infer_architecture_type(family_name, class_names_all, task_suffixes_set, modality)

    # Notes
    notes = []
    if has_modular:
        notes.append("Has modular file; modeling files are auto-generated")
    if status_flags["is_likely_shared_or_meta"]:
        notes.append("Likely shared infrastructure or meta module, not a standalone architecture")
    if status_flags["is_likely_legacy"]:
        notes.append("Likely deprecated/legacy")
    if not status_flags["has_pytorch"] and not status_flags["has_tensorflow"] and not status_flags["has_flax"]:
        notes.append("No modeling files found; may be tokenizer-only, config-only, or infrastructure")
    if len(task_head_classes) > 8:
        notes.append(f"Unusually broad head coverage ({len(task_head_classes)} task heads)")

    return {
        "family_name": family_name,
        "path": f"src/transformers/models/{family_name}",
        "status_flags": status_flags,
        "files": files_by_type,
        "classes": classes,
        "base_model_classes": base_model_classes,
        "task_head_classes": task_head_classes,
        "tasks_supported": tasks_supported,
        "task_head_suffixes": task_head_suffixes,
        "modality": modality,
        "architecture_type": arch_type,
        "notes": notes,
    }


def build_summary(families: list[dict]) -> dict:
    """Build aggregate summary from analyzed families."""
    family_count = len(families)
    all_classes = []
    for f in families:
        all_classes.extend(f["classes"])

    class_count = len(all_classes)
    base_model_count = sum(1 for c in all_classes if c["class_kind"] == "base_model")
    task_head_count = sum(1 for c in all_classes if c["class_kind"] == "task_head")

    modality_counter = Counter(f["modality"] for f in families)
    arch_counter = Counter(f["architecture_type"] for f in families)

    suffix_counter = Counter()
    for c in all_classes:
        if c["task_suffix"]:
            suffix_counter[c["task_suffix"]] += 1

    top_suffixes = [{"suffix": s, "count": c} for s, c in suffix_counter.most_common(30)]

    head_counts = [(f["family_name"], len(f["task_head_classes"])) for f in families if f["task_head_classes"]]
    head_counts.sort(key=lambda x: x[1], reverse=True)
    top_families_by_head = [{"family": name, "head_count": cnt} for name, cnt in head_counts[:20]]

    return {
        "family_count": family_count,
        "class_count": class_count,
        "base_model_class_count": base_model_count,
        "task_head_class_count": task_head_count,
        "families_by_modality": dict(modality_counter.most_common()),
        "families_by_architecture_type": dict(arch_counter.most_common()),
        "top_task_suffixes": top_suffixes,
        "top_families_by_head_count": top_families_by_head,
    }


def build_task_taxonomy(families: list[dict]) -> dict:
    """Build task taxonomy aggregating all families."""
    suffix_to_task = dict(SUFFIX_TO_TASK)
    task_to_families = defaultdict(list)
    task_to_classes = defaultdict(list)

    for fam in families:
        for c in fam["classes"]:
            if c["task_inferred"]:
                task_to_families[c["task_inferred"]].append(fam["family_name"])
                task_to_classes[c["task_inferred"]].append(c["class_name"])
                if c["task_suffix"] and c["task_suffix"] not in suffix_to_task:
                    suffix_to_task[c["task_suffix"]] = c["task_inferred"]

    # Deduplicate family lists
    task_to_families = {k: sorted(set(v)) for k, v in task_to_families.items()}
    task_to_classes = {k: sorted(set(v)) for k, v in task_to_classes.items()}

    return {
        "suffix_to_task": suffix_to_task,
        "task_to_families": dict(task_to_families),
        "task_to_classes": dict(task_to_classes),
    }


def build_cross_cutting(families: list[dict]) -> dict:
    """Build cross-cutting insights."""
    insights = {
        "families_with_causal_lm": [],
        "families_with_sequence_classification": [],
        "families_with_vision2seq": [],
        "families_with_conditional_generation": [],
        "families_with_audio_heads": [],
        "families_with_multimodal_processors": [],
    }

    for fam in families:
        suffixes = set(fam["task_head_suffixes"])
        if "ForCausalLM" in suffixes:
            insights["families_with_causal_lm"].append(fam["family_name"])
        if "ForSequenceClassification" in suffixes:
            insights["families_with_sequence_classification"].append(fam["family_name"])
        if "ForVision2Seq" in suffixes:
            insights["families_with_vision2seq"].append(fam["family_name"])
        if "ForConditionalGeneration" in suffixes:
            insights["families_with_conditional_generation"].append(fam["family_name"])
        audio_suffixes = {"ForCTC", "ForAudioClassification", "ForSpeechSeq2Seq", "ForAudioFrameClassification", "ForAudioXVector"}
        if suffixes & audio_suffixes:
            insights["families_with_audio_heads"].append(fam["family_name"])
        if fam["status_flags"]["has_processor"] and (fam["status_flags"]["has_image_processor"] or fam["modality"] == "multimodal"):
            insights["families_with_multimodal_processors"].append(fam["family_name"])

    return insights


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_markdown_report(data: dict, output_path: Path):
    """Generate the Markdown report."""
    summary = data["summary"]
    families = data["families"]
    taxonomy = data["task_taxonomy"]
    cross = data["cross_cutting_insights"]

    lines = []
    w = lines.append

    w("# Transformers Architecture Map Report")
    w("")
    w(f"**Generated:** {data['generated_at']}")
    w(f"**Repo root:** `{data['repo_root']}`")
    w(f"**Models path:** `{data['models_path']}`")
    w("")

    # 1. Executive summary
    w("## 1. Executive Summary")
    w("")
    w(f"| Metric | Count |")
    w(f"|--------|-------|")
    w(f"| Model family folders | {summary['family_count']} |")
    w(f"| Total classes parsed | {summary['class_count']} |")
    w(f"| Base model classes | {summary['base_model_class_count']} |")
    w(f"| Task head classes | {summary['task_head_class_count']} |")
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

    # 2. Family inventory (table)
    w("## 2. Family Inventory")
    w("")
    w(f"Showing all {len(families)} families. Sorted alphabetically.")
    w("")
    w("| Family | Modality | Architecture | Backends | Base Models | Task Heads | Tasks | Notes |")
    w("|--------|----------|--------------|----------|-------------|------------|-------|-------|")

    for fam in sorted(families, key=lambda f: f["family_name"]):
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

        # Escape pipes in cell content
        for s in (base_str, heads_str, tasks_str, notes_str):
            s = s.replace("|", "\\|")

        w(f"| {fam['family_name']} | {fam['modality']} | {fam['architecture_type']} | {backends_str} | {base_str} | {heads_str} | {tasks_str} | {notes_str} |")
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

    w("### Families with ForCausalLM (text-generation)")
    w("")
    causal_fams = cross["families_with_causal_lm"]
    w(f"**{len(causal_fams)} families:** {', '.join(sorted(causal_fams))}")
    w("")

    w("### Families with ForConditionalGeneration")
    w("")
    cond_fams = cross["families_with_conditional_generation"]
    w(f"**{len(cond_fams)} families:** {', '.join(sorted(cond_fams))}")
    w("")

    w("### Families with ForVision2Seq (image-to-text)")
    w("")
    v2s_fams = cross["families_with_vision2seq"]
    w(f"**{len(v2s_fams)} families:** {', '.join(sorted(v2s_fams))}")
    w("")

    w("### Likely Serving-Relevant Families (vLLM/SGLang)")
    w("")
    w("Families with ForCausalLM or ForConditionalGeneration that are decoder_only or encoder_decoder:")
    w("")
    serving_relevant = []
    for fam in families:
        suffixes = set(fam["task_head_suffixes"])
        if ("ForCausalLM" in suffixes or "ForConditionalGeneration" in suffixes) and fam["architecture_type"] in ("decoder_only", "encoder_decoder"):
            serving_relevant.append(fam["family_name"])
    w(f"**{len(serving_relevant)} families:** {', '.join(sorted(serving_relevant))}")
    w("")

    # 5. Backend coverage
    w("## 5. Backend Coverage")
    w("")
    pytorch_only = []
    pt_tf = []
    pt_flax = []
    all_three = []
    no_backend = []

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
        elif not sf["has_pytorch"] and not sf["has_tensorflow"] and not sf["has_flax"]:
            no_backend.append(fam["family_name"])

    w(f"| Backend Config | Count |")
    w(f"|----------------|-------|")
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

    w("### Likely Legacy/Deprecated")
    w("")
    for fam in families:
        if fam["status_flags"]["is_likely_legacy"]:
            w(f"- {fam['family_name']}")
    w("")

    # 7. Caveats
    w("## 7. Caveats")
    w("")
    w("- Folder count ≠ class count: one family can contain many classes and task heads.")
    w("- Modality and architecture type are heuristic-based and may misclassify specialized families.")
    w("- Some folders are infrastructure (e.g., `auto`), shared utilities, or deprecated wrappers.")
    w("- Class parsing uses Python AST; dynamically generated classes may be missed.")
    w("- 'is_auto_export_candidate' is a heuristic — actual AutoModel registration is defined in auto/.")
    w("- Modular files mean the modeling files are auto-generated and should not be edited directly.")
    w("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def generate_csv_family_summary(families: list[dict], output_path: Path):
    """Generate family summary CSV."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "family_name", "modality", "architecture_type",
            "has_pytorch", "has_tensorflow", "has_flax",
            "base_model_count", "task_head_count",
            "tasks_supported", "has_modular", "is_shared_meta", "is_legacy",
        ])
        for fam in sorted(families, key=lambda x: x["family_name"]):
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
            ])


def generate_csv_task_matrix(families: list[dict], output_path: Path):
    """Generate task matrix CSV: families x task suffixes."""
    # Collect all unique task suffixes
    all_suffixes = set()
    for fam in families:
        all_suffixes.update(fam["task_head_suffixes"])
    all_suffixes = sorted(all_suffixes)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["family_name"] + all_suffixes)
        for fam in sorted(families, key=lambda x: x["family_name"]):
            row = [fam["family_name"]]
            fam_suffixes = set(fam["task_head_suffixes"])
            for s in all_suffixes:
                row.append("X" if s in fam_suffixes else "")
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    print(f"Scanning model families in {MODELS_PATH} ...")

    if not MODELS_PATH.is_dir():
        print(f"ERROR: {MODELS_PATH} not found", file=sys.stderr)
        sys.exit(1)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Enumerate family directories
    family_dirs = sorted([
        d for d in MODELS_PATH.iterdir()
        if d.is_dir() and d.name != "__pycache__" and not d.name.startswith(".")
    ])

    print(f"Found {len(family_dirs)} family directories.")

    # Analyze each family
    families = []
    for fd in family_dirs:
        fam_data = analyze_family(fd)
        families.append(fam_data)

    # Build aggregates
    summary = build_summary(families)
    taxonomy = build_task_taxonomy(families)
    cross_cutting = build_cross_cutting(families)

    # Assemble final JSON
    data = {
        "repo_root": str(REPO_ROOT),
        "models_path": str(MODELS_PATH),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "families": families,
        "task_taxonomy": taxonomy,
        "cross_cutting_insights": cross_cutting,
    }

    # Write JSON
    json_path = ARTIFACTS_DIR / "transformers_arch_map.json"
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {json_path}")

    # Write Markdown
    md_path = ARTIFACTS_DIR / "transformers_arch_report.md"
    generate_markdown_report(data, md_path)
    print(f"Wrote {md_path}")

    # Write CSV files
    csv_family_path = ARTIFACTS_DIR / "transformers_family_summary.csv"
    generate_csv_family_summary(families, csv_family_path)
    print(f"Wrote {csv_family_path}")

    csv_task_path = ARTIFACTS_DIR / "transformers_task_matrix.csv"
    generate_csv_task_matrix(families, csv_task_path)
    print(f"Wrote {csv_task_path}")

    # Console summary
    print("\n" + "=" * 70)
    print("ARCHITECTURE MAP SUMMARY")
    print("=" * 70)
    print(f"  Total families:          {summary['family_count']}")
    print(f"  Total classes:           {summary['class_count']}")
    print(f"  Base model classes:      {summary['base_model_class_count']}")
    print(f"  Task head classes:       {summary['task_head_class_count']}")
    print()

    print("  Top 10 task head suffixes:")
    for item in summary["top_task_suffixes"][:10]:
        task = SUFFIX_TO_TASK.get(item["suffix"], item["suffix"])
        print(f"    {item['suffix']:40s} {item['count']:4d}  ({task})")
    print()

    print("  Top 10 families by head count:")
    for item in summary["top_families_by_head_count"][:10]:
        print(f"    {item['family']:40s} {item['head_count']:4d}")
    print()

    print("  Modality distribution:")
    for mod, cnt in sorted(summary["families_by_modality"].items(), key=lambda x: -x[1]):
        print(f"    {mod:20s} {cnt:4d}")
    print()

    print("  Architecture type distribution:")
    for atype, cnt in sorted(summary["families_by_architecture_type"].items(), key=lambda x: -x[1]):
        print(f"    {atype:20s} {cnt:4d}")
    print()

    causal_count = len(cross_cutting["families_with_causal_lm"])
    cond_count = len(cross_cutting["families_with_conditional_generation"])
    serving_count = len([
        f for f in families
        if (set(f["task_head_suffixes"]) & {"ForCausalLM", "ForConditionalGeneration"})
        and f["architecture_type"] in ("decoder_only", "encoder_decoder")
    ])
    print(f"  Families with ForCausalLM:            {causal_count}")
    print(f"  Families with ForConditionalGeneration:{cond_count}")
    print(f"  Likely LLM-serving-relevant families:  {serving_count}")
    print("=" * 70)


if __name__ == "__main__":
    main()
