"""
Layer 1: Universal Model-Agnostic Semantic Asset Layer (Immortal IR & Versioned Registry)
========================================================================================
Follows SYSTEM_CONSTITUTION.md [Tier-A] Production Core:
- Stores model-agnostic cognitive primitives (programs, AST templates, preconditions,
  postconditions, provenance, counterfactual test suites, counterexamples, version history).
- Supports soft-deprecation (quarantine) for rare high-value skills instead of blind deletion,
  plus deterministic one-click rollback to any prior snapshot.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class VerificationRecord:
    timestamp: float
    passed: bool
    ood_accuracy: float
    base_accuracy: float
    non_target_drift: float
    amortized_speedup: float
    verifier_type: str
    notes: str = ""


@dataclass
class CognitivePrimitiveIR:
    """Model-agnostic immortal intermediate representation of a verified cognitive skill."""
    primitive_id: str
    name: str
    domain: str
    version: int
    code_source: str
    ast_template: str
    preconditions: List[str]
    postconditions: List[str]
    input_schema: Dict[str, str]
    provenance_task_ids: List[str]
    unit_tests: List[Dict[str, Any]]
    counterexamples: List[Dict[str, Any]] = field(default_factory=list)
    verification_history: List[VerificationRecord] = field(default_factory=list)
    status: str = "active"  # active | quarantined | rolled_back
    rarity_critical: bool = False  # Protects rare high-value skills from frequency-based eviction
    usage_count: int = 0
    success_count: int = 0
    created_at: float = field(default_factory=time.time)

    def content_hash(self) -> str:
        payload = json.dumps(
            {
                "name": self.name,
                "domain": self.domain,
                "code_source": self.code_source,
                "preconditions": self.preconditions,
                "postconditions": self.postconditions,
            },
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]


class ImmortalIRRegistry:
    """Persistent, versioned, rollback-capable registry for Layer 1 Semantic Assets."""

    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.primitives: Dict[str, CognitivePrimitiveIR] = {}
        self.snapshots: List[Dict[str, Any]] = []
        self.audit_log: List[Dict[str, Any]] = []
        if os.path.dirname(storage_path):
            os.makedirs(os.path.dirname(storage_path), exist_ok=True)
        if os.path.exists(storage_path):
            self.load()

    def _take_snapshot(self, reason: str) -> int:
        snap_id = len(self.snapshots)
        self.snapshots.append(
            {
                "snapshot_id": snap_id,
                "timestamp": time.time(),
                "reason": reason,
                "state": copy.deepcopy({k: asdict(v) for k, v in self.primitives.items()}),
            }
        )
        return snap_id

    def register_primitive(self, primitive: CognitivePrimitiveIR, reason: str = "register") -> int:
        snap_id = self._take_snapshot(f"before_{reason}:{primitive.primitive_id}")
        if primitive.primitive_id in self.primitives:
            primitive.version = self.primitives[primitive.primitive_id].version + 1
        self.primitives[primitive.primitive_id] = primitive
        self.audit_log.append(
            {
                "time": time.time(),
                "action": reason,
                "primitive_id": primitive.primitive_id,
                "version": primitive.version,
                "hash": primitive.content_hash(),
                "rollback_snapshot": snap_id,
            }
        )
        self.save()
        return snap_id

    def soft_quarantine(self, primitive_id: str, reason: str) -> bool:
        """Soft-deprecation instead of destructive deletion (Constitution Rule)."""
        if primitive_id not in self.primitives:
            return False
        prim = self.primitives[primitive_id]
        if prim.rarity_critical:
            self.audit_log.append(
                {
                    "time": time.time(),
                    "action": "quarantine_blocked_rarity_critical",
                    "primitive_id": primitive_id,
                    "reason": reason,
                }
            )
            self.save()
            return False
        self._take_snapshot(f"before_quarantine:{primitive_id}")
        prim.status = "quarantined"
        self.audit_log.append(
            {
                "time": time.time(),
                "action": "soft_quarantine",
                "primitive_id": primitive_id,
                "reason": reason,
            }
        )
        self.save()
        return True

    def rollback_to_snapshot(self, snapshot_id: int) -> bool:
        if snapshot_id < 0 or snapshot_id >= len(self.snapshots):
            return False
        snap = self.snapshots[snapshot_id]
        restored: Dict[str, CognitivePrimitiveIR] = {}
        for k, v in snap["state"].items():
            v_copy = copy.deepcopy(v)
            v_copy["verification_history"] = [
                VerificationRecord(**vr) for vr in v_copy.get("verification_history", [])
            ]
            restored[k] = CognitivePrimitiveIR(**v_copy)
        self.primitives = restored
        self.audit_log.append(
            {
                "time": time.time(),
                "action": "rollback",
                "target_snapshot_id": snapshot_id,
            }
        )
        self.save()
        return True

    def list_active(self, domain: Optional[str] = None) -> List[CognitivePrimitiveIR]:
        res = [p for p in self.primitives.values() if p.status == "active"]
        if domain is not None:
            res = [p for p in res if p.domain == domain]
        return res

    def save(self) -> None:
        payload = {
            "primitives": {k: asdict(v) for k, v in self.primitives.items()},
            "audit_log": self.audit_log,
            "snapshot_count": len(self.snapshots),
        }
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def load(self) -> None:
        with open(self.storage_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.primitives = {}
        for k, v in payload.get("primitives", {}).items():
            v["verification_history"] = [
                VerificationRecord(**vr) for vr in v.get("verification_history", [])
            ]
            self.primitives[k] = CognitivePrimitiveIR(**v)
        self.audit_log = payload.get("audit_log", [])
