"""Small heading/keyword/rule retriever; no vector database required."""

from __future__ import annotations

import re
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


METHOD_FILE_NAMES = {
    "matrix": "matrix_factorization.md",
    "mode3": "mode3_factorization.md",
    "cp": "cp_decomposition.md",
    "nonnegative_cp": "nonnegative_cp_decomposition.md",
    "tucker": "tucker_decomposition.md",
    "btd": "block_term_decomposition.md",
    "tsvd": "t_svd_decomposition.md",
    "nonnegative_tucker": "nonnegative_tucker_decomposition.md",
    "hierarchical_tucker": "hierarchical_tucker_decomposition.md",
    "tt": "tensor_train_decomposition.md",
    "tensor_ring": "tensor_ring_decomposition.md",
}


def canonical_data_type(profile: Dict[str, Any]) -> str:
    """Honor semantic metadata: framed audio must not be inferred as MSI."""
    value = profile.get("data_type") or (profile.get("source_metadata") or {}).get("data_type")
    aliases = {"image": "Image", "color_image": "Image", "msi": "MSI", "video": "Video", "audio": "audio"}
    if value:
        if str(value).lower() not in aliases:
            raise ValueError("unknown knowledge data type %r" % value)
        return aliases[str(value).lower()]
    shape = profile.get("image_shape", [])
    return "Video" if len(shape) == 4 else "Image" if len(shape) == 3 and shape[-1] == 3 else "MSI"


def selection_context(profile: Dict[str, Any]) -> Dict[str, Any]:
    kind = canonical_data_type(profile)
    shape = profile.get("image_shape", [])
    features = math.prod(shape[2:]) if len(shape) >= 3 else profile.get("feature_count")
    warnings = [
        "Applicable conditions are engineering hypotheses, not measured rankings or recovery guarantees.",
        "Correlation, preview frequency and 2-D projected hole statistics are coarse proxies, not measured tensor ranks.",
        "Train only on observations; GT missing-region feedback selects checkpoints/configurations. Whole-modality means decide recovery screening/promotion; audio uses only waveform NMSE.",
    ]
    if kind == "audio":
        warnings.extend(["Frame index and within-frame sample are time representation axes, not image spatial axes; joint feature mode is channels.",
                         "Mono t-SVD FFT is length one (not temporal FFT/STFT); mono mode3 has no temporal coupling.",
                         "Normalized nonnegative values do not justify nonnegative latent factors for the original signed waveform."])
    elif kind == "Video":
        warnings.append("Tensor baselines use H x W x (T*C), not four independent modes; color-video t-SVD FFT mixes time and color.")
    elif kind == "MSI":
        warnings.append("Feature rank is bounded by band count, not the RGB limit of three; wavelength order/smoothness is not proven by channel correlation.")
    pattern = profile.get("mask_type")
    if pattern in {"slices", "sildes"}:
        warnings.append("Entire slices can leave free-index factors without data constraints; low rank alone does not guarantee slice recovery. Recovery axes: Image rows, MSI bands, Video frames; audio slices are continuous waveform gaps.")
    rate = profile.get("actual_missing_rate")
    if rate is not None and float(rate) >= .7:
        warnings.append("High missingness (heuristic >=0.7) requires checking observation coverage and parameter budget; it does not determine an optimal rank/family.")
    if profile.get("structure_statistics_imputation"):
        warnings.append("Structure statistics include visible-feature-mean imputation, not actual observations for every feature; 2-D holes do not describe full spectral/time slices.")
    return {"data_type": kind, "model_view": "D1 x D2 x F", "feature_count": features,
            "axis_semantics": ["time_frame", "sample_in_frame", "channel"] if kind == "audio" else
                ["height", "width", "time*channel" if kind == "Video" else "band" if kind == "MSI" else "channel"],
            "mask_type": "slices" if pattern == "sildes" else pattern,
            "actual_missing_rate": rate, "warnings": warnings}


def _condition_is_active(condition: str, profile: Dict[str, Any]) -> bool:
    kind = canonical_data_type(profile)
    shape = profile.get("image_shape", [])
    features = math.prod(shape[2:]) if len(shape) >= 3 else int(profile.get("feature_count") or 0)
    def number(key):
        value = profile.get(key)
        return float(value) if value is not None else math.nan
    correlation = number("visible_mean_absolute_channel_correlation")
    largest_hole = number("largest_missing_component_image_ratio")
    component_count = number("missing_component_count")
    spatial = kind != "audio"
    multi_feature = spatial and features >= 2
    condition_map = {
        "block_missing": profile["mask_type"] == "block",
        "random_missing": profile["mask_type"] == "random",
        "channel_correlation_high": multi_feature and correlation >= 0.75,
        "channel_correlation_low": multi_feature and correlation < 0.35,
        "large_contiguous_hole": spatial and largest_hole >= 0.20,
        "many_small_holes": spatial and component_count >= 8 and largest_hole < 0.10,
        "image_aspect_ratio_high": spatial and number("image_aspect_ratio") >= 1.6,
        "local_smoothness_high": spatial and number("visible_local_smoothness_score") >= 0.75,
        "high_frequency_high": spatial and number("visible_high_frequency_energy_ratio") >= 0.08,
        "slices_missing": profile["mask_type"] in {"slices", "sildes"},
        "feature_mode_large": features > 3,
        "ordered_feature_mode": (kind == "MSI" and features > 3 or kind == "Video" and len(shape) == 4 and shape[-1] == 1 and shape[2] > 1),
        "audio_mono": kind == "audio" and features == 1,
        "data_type_msi": kind == "MSI", "data_type_video": kind == "Video", "data_type_audio": kind == "audio",
    }
    if condition not in condition_map:
        raise ValueError("unknown selection condition %r" % condition)
    return condition_map[condition]


def _markdown_chunks(path: Path) -> List[Dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    chunks = []
    heading = "Introduction"
    body_lines = []
    for line in text.splitlines():
        if line.startswith("#"):
            if body_lines:
                chunks.append(
                    {
                        "heading": heading,
                        "content": "\n".join(body_lines).strip(),
                    }
                )
            heading = line.lstrip("#").strip()
            body_lines = []
        else:
            body_lines.append(line)
    if body_lines:
        chunks.append(
            {"heading": heading, "content": "\n".join(body_lines).strip()}
        )
    return [chunk for chunk in chunks if chunk["content"]]


class LocalKnowledgeRetriever:
    """Retrieve auditable evidence from local Markdown and YAML rules."""

    def __init__(
        self,
        knowledge_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.knowledge_dir = (
            Path(knowledge_dir)
            if knowledge_dir is not None
            else Path(__file__).resolve().parent
        )
        self.methods_dir = self.knowledge_dir / "methods"
        self.rules_path = self.knowledge_dir / "selection_rules.yaml"
        if not self.rules_path.is_file():
            raise ValueError("selection rules do not exist: %s" % self.rules_path)

    def active_rules(self, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        payload = yaml.safe_load(self.rules_path.read_text(encoding="utf-8"))
        rules = payload.get("rules", []) if isinstance(payload, dict) else []
        active = []
        for index, rule in enumerate(rules):
            kinds = rule.get("data_types")
            if kinds is not None and (not isinstance(kinds, list) or any(kind not in {"Image", "MSI", "Video", "audio"} for kind in kinds)):
                raise ValueError("selection rule data_types must be a list of supported types")
            if kinds is not None and canonical_data_type(profile) not in kinds:
                continue
            if _condition_is_active(str(rule["condition"]), profile):
                active.append(
                    {
                        **rule,
                        "source": "selection_rules.yaml#rule-%d" % index,
                    }
                )
        return active

    def retrieve(
        self,
        profile: Dict[str, Any],
        query: str,
        top_k: int = 8,
    ) -> Dict[str, Any]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        active_rules = self.active_rules(profile)
        query_terms = set(re.findall(r"[a-zA-Z0-9_]+", query.lower()))
        for rule in active_rules:
            query_terms.add(str(rule["prefer"]).lower())
            for keyword in rule.get("keywords", []):
                query_terms.update(
                    re.findall(r"[a-zA-Z0-9_]+", str(keyword).lower())
                )

        evidence = []
        catalogue = []
        kind = canonical_data_type(profile)
        preferred_methods = [str(rule["prefer"]) for rule in active_rules]
        for method, filename in METHOD_FILE_NAMES.items():
            path = self.methods_dir / filename
            for chunk in _markdown_chunks(path):
                modality = re.match(r"^(Image|MSI|Video|audio) applicability$", chunk["heading"], re.I)
                if modality:
                    if modality.group(1).lower() != kind.lower():
                        continue
                    catalogue.append({"source": "%s#%s" % (filename, chunk["heading"]),
                                      "method": method, "heading": chunk["heading"], "content": chunk["content"]})
                    continue  # Included once in the all-family catalogue, not again as a deeper chunk.
                if chunk["heading"] == "Evidence boundary":
                    continue  # Bibliography is not an empirical reason to prefer a family.
                searchable = (method + "\n" + chunk["heading"] + "\n" + chunk["content"]).lower()
                score = sum(1.0 for term in query_terms if term in searchable)
                score += 2.0 * preferred_methods.count(method)
                if score <= 0.0:
                    continue
                evidence.append(
                    {
                        "source": "%s#%s" % (filename, chunk["heading"]),
                        "method": method,
                        "heading": chunk["heading"],
                        "score": score,
                        "content": chunk["content"],
                    }
                )
        evidence.sort(key=lambda item: (-item["score"], item["source"]))
        selected = []
        seen = set()
        for item in evidence:
            if item["method"] not in seen:
                selected.append(item)
                seen.add(item["method"])
                if len(selected) == top_k:
                    break
        selected_sources = {item["source"] for item in selected}
        selected.extend(item for item in evidence if item["source"] not in selected_sources)
        return {
            "query": query,
            "active_rules": active_rules,
            "evidence": selected[:top_k],
            "method_catalogue": catalogue,
            "selection_context": selection_context(profile),
        }
