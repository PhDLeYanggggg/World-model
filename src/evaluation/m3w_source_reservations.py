"""Preserve whole-source holdouts before predictive admission.

This is an accidental-use guard, not a filesystem sandbox or an independence
certificate. Matching roles still need source review and the experiment contract.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

from src.evaluation.m3w_recording_lineage import sha256


REGISTRY = "configs/m3w_external_source_reservations_v1.json"
KIND = "m3w_whole_source_reservations_v1"
PROTECTED_DATASETS = {"dut", "dronecrowd"}
PROTECTED_PREFIXES = (
    "data/stage_cvpr2027_causal/dut_diagnostic",
    "data/stage_cvpr2027_experiments/dronecrowd_recordings_v1",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def contained(root, relative):
    require(isinstance(relative, str) and relative, "Missing reservation path")
    path = (root / relative).resolve()
    require(path.is_relative_to(root), "Reservation path escapes workspace")
    return path


def metadata_identities(metadata):
    values = {metadata.get("source_xml_sha256")}
    values.update(item.get("sha256") for item in metadata.get("files", []))
    # ID vectors and all-true masks can legitimately be identical across clips.
    # Only geometry payloads can identify renamed recordings here.
    values.update(item.get("sha256") for name, item in metadata.get("artifacts", {}).items()
                  if name in {"points.npy", "positions.npy"})
    return {value for value in values if isinstance(value, str) and len(value) == 64}


class SourceReservations:
    def __init__(self, root):
        self.root = Path(root).resolve()
        path = self.root / REGISTRY
        self.bindings = {REGISTRY: sha256(path)}
        self.document = json.loads(path.read_text())
        require(self.document.get("kind") == KIND, "Unsupported source reservation schema")
        require(self.document.get("status") == "frozen_reservations_not_predictive_admission",
                "Reservation cannot grant predictive admission")
        require(self.document.get("independence_certified") is False,
                "Reservation does not certify physical-site independence")
        for relative, digest in self.document["bindings"].items():
            evidence = contained(self.root, relative)
            require(evidence.is_file() and sha256(evidence) == digest,
                    f"Changed reservation evidence: {relative}")
            self.bindings[relative] = digest
        self.records = self.document["records"]
        require(isinstance(self.records, dict) and self.records, "Empty source reservation")
        for name, record in self.records.items():
            require(record["reserved_role"] in {"calibration", "confirmation", "excluded"},
                    "Holdout cannot be reserved for fitting or model selection")
            require(record.get("predictive_admission") is False, "Unexpected predictive reservation grant")
            require(isinstance(record.get("source_identities"), list) and record["source_identities"],
                    "Source identities required for renamed-copy protection")
            contained(self.root, record["cache_path"])
            require(re.fullmatch("[0-9a-f]{64}", record.get("metadata_sha256", ""))
                    and all(isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value)
                            for value in record["source_identities"]), "Invalid reserved identity digest")
        # Whole-source closure is stricter than the observed pairwise links.
        for dataset, source in self.document["sources"].items():
            rows = [r for r in self.records.values() if r["dataset"] == dataset]
            require(rows and all(r["reserved_role"] in {source["reserved_role"], "excluded"} for r in rows),
                    "A reserved source crosses predictive roles")
            names = {n for n, r in self.records.items() if r["dataset"] == dataset}
            require(names == set(source["recordings"]), "Incomplete source reservation coverage")
        require({r["dataset"] for r in self.records.values()} == set(self.document["sources"]),
                "Reserved recording has no source-wide declaration")

    def validate(self, name, record, metadata, role):
        """Return evidence bindings; successful matching is not admission."""
        path = contained(self.root, record["cache_path"])
        identities = metadata_identities(metadata)
        dataset = str(metadata.get("dataset", "")).casefold()
        matches = [(key, item) for key, item in self.records.items()
                   if path == contained(self.root, item["cache_path"])
                   or record.get("metadata_sha256") == item["metadata_sha256"]
                   or bool(identities.intersection(item["source_identities"]))]
        protected = (dataset in PROTECTED_DATASETS
                     or metadata.get("schema") == "m3w_dronecrowd_recording_cache_v1"
                     or any(path.is_relative_to(contained(self.root, prefix)) for prefix in PROTECTED_PREFIXES))
        if not matches:
            require(not protected, "Protected source absent from frozen reservation inventory")
            return {}
        require(role != "excluded", "Excluded records do not require predictive admission")
        require(all(item["reserved_role"] == role for _, item in matches),
                "Whole-source reservation forbids this data role (including aliases)")
        require(len(matches) == 1 and matches[0][0] == name
                and path == contained(self.root, matches[0][1]["cache_path"])
                and record.get("metadata_sha256") == matches[0][1]["metadata_sha256"],
                "Reserved source must use its exact reviewed recording identity")
        # Unrelated portable runs must not require the protected raw/cache tree.
        # Check cache bytes only for a matching, correctly named reserved source.
        metadata_path = path / "metadata.json"
        require(metadata_path.is_file() and sha256(metadata_path) == record["metadata_sha256"],
                "Changed reserved cache metadata")
        require(metadata == json.loads(metadata_path.read_text())
                and sorted(identities) == matches[0][1]["source_identities"],
                "Reserved source metadata disagrees with the reviewed cache")
        return dict(self.bindings)


def check_source_reservation(root, name, record, metadata, role):
    if role == "excluded":
        return {}
    root = Path(root).resolve()
    if (root / REGISTRY).exists():
        return SourceReservations(root).validate(name, record, metadata, role)
    dataset = str(metadata.get("dataset", "")).casefold()
    path = contained(root, record["cache_path"])
    protected = (dataset in PROTECTED_DATASETS
                 or metadata.get("schema") == "m3w_dronecrowd_recording_cache_v1"
                 or any(path.is_relative_to(contained(root, prefix)) for prefix in PROTECTED_PREFIXES))
    require(not protected, "Protected source requires a frozen reservation inventory")
    return {}
