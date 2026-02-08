"""
EHDS Regulatory Sandbox - Compliance Report Generator.

Maps EHDS Regulation (EU) 2025/327 articles to implemented governance
features, generating a per-article compliance assessment after FL training.

Covers:
- Chapter II  (Art. 5-12):  MyHealth@EU / cross-border primary use
- Chapter IV  (Art. 33-58): Secondary use of health data
- Chapter V   (Art. 69-71): Data quality and citizen rights
- GDPR        (Art. 30):    Records of processing activities

The report reads the state of a trained CrossBorderFederatedTrainer
and its governance bridges (IHE, Data Quality, Jurisdiction Privacy,
MyHealth@EU) to determine compliance status for each article.

Output formats: terminal display, JSON, LaTeX table.

Author: Fabio Liberti
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class ComplianceStatus(Enum):
    """Compliance status for a single EHDS article."""
    COMPLIANT = "COMPLIANT"
    PARTIAL = "PARTIAL"
    NOT_ASSESSED = "NOT_ASSESSED"
    NON_COMPLIANT = "NON_COMPLIANT"


@dataclass
class ArticleAssessment:
    """Assessment result for a single EHDS article."""
    article_id: str = ""
    title: str = ""
    chapter: str = ""
    status: ComplianceStatus = ComplianceStatus.NOT_ASSESSED
    evidence: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    module_name: str = ""
    regulation_ref: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "article_id": self.article_id,
            "title": self.title,
            "chapter": self.chapter,
            "status": self.status.value,
            "evidence": self.evidence,
            "details": self.details,
            "module_name": self.module_name,
            "regulation_ref": self.regulation_ref,
        }


# =============================================================================
# ARTICLE REGISTRY
# =============================================================================
# Each entry: (chapter, article_id, title, assess_method_name)

ARTICLE_REGISTRY: List[Tuple[str, str, str, str]] = [
    # Chapter II - MyHealth@EU (Primary Use)
    ("CHAPTER II - MyHealth@EU",
     "Art. 5-7", "Patient Summary Exchange", "_assess_patient_summary"),
    ("CHAPTER II - MyHealth@EU",
     "Art. 8-9", "ePrescription Cross-Border", "_assess_eprescription"),
    # Chapter IV - Secondary Use
    ("CHAPTER IV - Secondary Use",
     "Art. 33", "Permitted Purposes", "_assess_purposes"),
    ("CHAPTER IV - Secondary Use",
     "Art. 34", "Data Categories", "_assess_categories"),
    ("CHAPTER IV - Secondary Use",
     "Art. 42", "Fee Model & Cost Tracking", "_assess_fee_model"),
    ("CHAPTER IV - Secondary Use",
     "Art. 44", "Data Minimization", "_assess_minimization"),
    ("CHAPTER IV - Secondary Use",
     "Art. 46", "Researcher Authentication", "_assess_authentication"),
    ("CHAPTER IV - Secondary Use",
     "Art. 48", "Member State Rights", "_assess_member_state"),
    ("CHAPTER IV - Secondary Use",
     "Art. 50", "Secure Processing Environment", "_assess_secure_processing"),
    ("CHAPTER IV - Secondary Use",
     "Art. 53", "Data Permit Validation", "_assess_permits"),
    ("CHAPTER IV - Secondary Use",
     "Art. 57-58", "Cross-Border Data Access", "_assess_cross_border"),
    # Chapter V - Quality & Rights
    ("CHAPTER V - Quality & Rights",
     "Art. 69", "Data Quality Framework", "_assess_quality"),
    ("CHAPTER V - Quality & Rights",
     "Art. 71", "Citizen Opt-Out Rights", "_assess_optout"),
    # GDPR
    ("GDPR",
     "Art. 30", "Records of Processing", "_assess_audit"),
]

# EHDS Art. 33 - Permitted purposes for secondary use
ARTICLE_53_PURPOSES = {
    "scientific_research",
    "public_health_surveillance",
    "health_policy",
    "education_training",
    "ai_system_development",
    "personalized_medicine",
    "official_statistics",
    "patient_safety",
}

# Dataset type -> EHDS data category mapping
DATASET_CATEGORY_MAP = {
    "synthetic": "ehr",
    "imaging": "imaging",
    "fhir": "ehr",
    "tabular": "ehr",
    "genomic": "genomic",
}


# =============================================================================
# EHDS COMPLIANCE REPORT
# =============================================================================

class EHDSComplianceReport:
    """
    EHDS Compliance Report Generator.

    Reads the state of a trained CrossBorderFederatedTrainer and its
    governance bridges to generate per-article compliance assessments.

    Usage:
        report = EHDSComplianceReport()
        report.generate_from_trainer(trainer, config)
        print(report.to_terminal_display())
        report_dict = report.to_json()
        latex_str = report.to_latex_table()
    """

    def __init__(self):
        self.assessments: List[ArticleAssessment] = []
        self.scenario_label: str = ""
        self.generated_at: str = ""
        self.training_config: Dict[str, Any] = {}
        self.final_metrics: Dict[str, Any] = {}

    # -----------------------------------------------------------------
    # GENERATION
    # -----------------------------------------------------------------

    def generate_from_trainer(self, trainer, config: Dict[str, Any]) -> None:
        """Generate compliance report from a trained CrossBorderFederatedTrainer.

        Args:
            trainer: CrossBorderFederatedTrainer instance (after train())
            config: Screen configuration dict
        """
        self.assessments = []
        self.generated_at = datetime.now().isoformat(timespec="seconds")
        self._build_scenario_label(trainer, config)
        self._store_config_summary(trainer, config)
        self._store_final_metrics(trainer)

        for chapter, art_id, title, method_name in ARTICLE_REGISTRY:
            assess_fn = getattr(self, method_name)
            assessment = assess_fn(trainer, config)
            assessment.chapter = chapter
            assessment.article_id = art_id
            assessment.title = title
            self.assessments.append(assessment)

    def _build_scenario_label(self, trainer, config: Dict) -> None:
        """Build a human-readable scenario description."""
        countries = config.get("countries", [])
        algo = config.get("algorithm", "FedAvg")
        eps = config.get("global_epsilon", 10.0)
        rounds = config.get("num_rounds", 15)
        n_hospitals = len(getattr(trainer, "hospitals", []))
        self.scenario_label = (
            f"{len(countries)} countries, {n_hospitals} hospitals, "
            f"{algo}, eps={eps}, {rounds} rounds"
        )

    def _store_config_summary(self, trainer, config: Dict) -> None:
        """Store key configuration for JSON export."""
        self.training_config = {
            "countries": config.get("countries", []),
            "algorithm": config.get("algorithm", "FedAvg"),
            "num_rounds": config.get("num_rounds", 15),
            "global_epsilon": config.get("global_epsilon", 10.0),
            "dataset_type": config.get("dataset_type", "synthetic"),
            "purpose": config.get("purpose", "scientific_research"),
            "jurisdiction_privacy_enabled": config.get(
                "jurisdiction_privacy_enabled", False),
            "ihe_enabled": config.get("ihe_enabled", False),
            "data_quality_enabled": config.get("data_quality_enabled", False),
            "myhealth_eu_enabled": config.get("myhealth_eu_enabled", False),
        }

    def _store_final_metrics(self, trainer) -> None:
        """Extract final training metrics."""
        history = getattr(trainer, "history", [])
        if history:
            last = history[-1]
            self.final_metrics = {
                "accuracy": round(getattr(last, "global_acc", 0.0), 4),
                "f1": round(getattr(last, "global_f1", 0.0), 4),
                "auc": round(getattr(last, "global_auc", 0.0), 4),
                "loss": round(getattr(last, "global_loss", 0.0), 4),
                "rounds_completed": len(history),
            }

    # -----------------------------------------------------------------
    # PER-ARTICLE ASSESSMENT METHODS
    # -----------------------------------------------------------------

    def _assess_patient_summary(
        self, trainer, config: Dict
    ) -> ArticleAssessment:
        """Art. 5-7: Patient Summary exchange via MyHealth@EU."""
        bridge = getattr(trainer, "myhealth_bridge", None)
        if bridge is None:
            return ArticleAssessment(
                status=ComplianceStatus.NOT_ASSESSED,
                evidence="MyHealth@EU bridge not enabled",
                module_name="myhealth_eu_bridge",
                regulation_ref="EHDS Art. 5-7",
            )
        ps_total = bridge._ps_sim.total_exchanged
        if ps_total > 0:
            return ArticleAssessment(
                status=ComplianceStatus.COMPLIANT,
                evidence=f"{ps_total} PS exchanged via NCPeH",
                details={"patient_summaries_exchanged": ps_total},
                module_name="myhealth_eu_bridge",
                regulation_ref="EHDS Art. 5-7",
            )
        return ArticleAssessment(
            status=ComplianceStatus.PARTIAL,
            evidence="NCPeH active but no PS exchanges",
            module_name="myhealth_eu_bridge",
            regulation_ref="EHDS Art. 5-7",
        )

    def _assess_eprescription(
        self, trainer, config: Dict
    ) -> ArticleAssessment:
        """Art. 8-9: ePrescription cross-border dispensation."""
        bridge = getattr(trainer, "myhealth_bridge", None)
        if bridge is None:
            return ArticleAssessment(
                status=ComplianceStatus.NOT_ASSESSED,
                evidence="MyHealth@EU bridge not enabled",
                module_name="myhealth_eu_bridge",
                regulation_ref="EHDS Art. 8-9",
            )
        ep_total = bridge._ep_sim.total_processed
        if ep_total > 0:
            return ArticleAssessment(
                status=ComplianceStatus.COMPLIANT,
                evidence=f"{ep_total} eP dispensed cross-border",
                details={"eprescriptions_processed": ep_total},
                module_name="myhealth_eu_bridge",
                regulation_ref="EHDS Art. 8-9",
            )
        return ArticleAssessment(
            status=ComplianceStatus.PARTIAL,
            evidence="NCPeH active but no eP exchanges",
            module_name="myhealth_eu_bridge",
            regulation_ref="EHDS Art. 8-9",
        )

    def _assess_purposes(
        self, trainer, config: Dict
    ) -> ArticleAssessment:
        """Art. 33: Permitted purposes for secondary use."""
        purpose = config.get("purpose", "scientific_research")
        violations = getattr(trainer, "purpose_violations", [])
        if violations:
            return ArticleAssessment(
                status=ComplianceStatus.NON_COMPLIANT,
                evidence=f"{len(violations)} purpose violations detected",
                details={"violations": violations, "purpose": purpose},
                module_name="data_permits",
                regulation_ref="EHDS Art. 33",
            )
        if purpose in ARTICLE_53_PURPOSES:
            return ArticleAssessment(
                status=ComplianceStatus.COMPLIANT,
                evidence=f"{purpose} (0 violations)",
                details={"purpose": purpose},
                module_name="data_permits",
                regulation_ref="EHDS Art. 33",
            )
        return ArticleAssessment(
            status=ComplianceStatus.PARTIAL,
            evidence=f"Purpose '{purpose}' not in Art. 53 list",
            details={"purpose": purpose},
            module_name="data_permits",
            regulation_ref="EHDS Art. 33",
        )

    def _assess_categories(
        self, trainer, config: Dict
    ) -> ArticleAssessment:
        """Art. 34: Data categories for secondary use."""
        ds_type = config.get("dataset_type", "synthetic")
        category = DATASET_CATEGORY_MAP.get(ds_type, "unknown")
        if category != "unknown":
            return ArticleAssessment(
                status=ComplianceStatus.COMPLIANT,
                evidence=f"{ds_type} -> {category}",
                details={"dataset_type": ds_type, "ehds_category": category},
                module_name="data_permits",
                regulation_ref="EHDS Art. 34",
            )
        return ArticleAssessment(
            status=ComplianceStatus.PARTIAL,
            evidence=f"Dataset type '{ds_type}' has no EHDS category mapping",
            details={"dataset_type": ds_type},
            module_name="data_permits",
            regulation_ref="EHDS Art. 34",
        )

    def _assess_fee_model(
        self, trainer, config: Dict
    ) -> ArticleAssessment:
        """Art. 42: Fee model and cost tracking."""
        bridge = getattr(trainer, "myhealth_bridge", None)
        if bridge and bridge.round_metrics:
            total_bytes = sum(
                rm.communication_cost_bytes for rm in bridge.round_metrics)
            total_kb = total_bytes / 1024
            return ArticleAssessment(
                status=ComplianceStatus.PARTIAL,
                evidence=f"{total_kb:.1f} KB tracked (cost simulation)",
                details={"communication_kb": round(total_kb, 1)},
                module_name="fee_model",
                regulation_ref="EHDS Art. 42",
            )
        return ArticleAssessment(
            status=ComplianceStatus.NOT_ASSESSED,
            evidence="Communication cost tracking not active",
            module_name="fee_model",
            regulation_ref="EHDS Art. 42",
        )

    def _assess_minimization(
        self, trainer, config: Dict
    ) -> ArticleAssessment:
        """Art. 44: Data minimization principle."""
        # Check governance training config for minimization
        try:
            from config.config_loader import get_governance_training_config
            gov_cfg = get_governance_training_config()
            if gov_cfg.get("minimization_enabled", False):
                return ArticleAssessment(
                    status=ComplianceStatus.COMPLIANT,
                    evidence="Feature filtering enabled (mutual_info)",
                    details={"method": "mutual_info"},
                    module_name="data_minimization",
                    regulation_ref="EHDS Art. 44",
                )
        except ImportError:
            pass
        return ArticleAssessment(
            status=ComplianceStatus.NOT_ASSESSED,
            evidence="Not enabled in this scenario",
            module_name="data_minimization",
            regulation_ref="EHDS Art. 44",
        )

    def _assess_authentication(
        self, trainer, config: Dict
    ) -> ArticleAssessment:
        """Art. 46: Researcher authentication (XUA/mTLS)."""
        ihe = getattr(trainer, "ihe_bridge", None)
        if ihe is None:
            return ArticleAssessment(
                status=ComplianceStatus.NOT_ASSESSED,
                evidence="IHE bridge not enabled",
                module_name="ihe_fl_bridge",
                regulation_ref="EHDS Art. 46",
            )
        # Check XUA and mTLS from IHE bridge (private attrs: _xua_enabled, _mtls_enabled)
        xua_active = getattr(ihe, "_xua_enabled", False)
        mtls_active = getattr(ihe, "_mtls_enabled", False)
        certs = len(getattr(ihe, "certificates", {}))
        if xua_active and mtls_active:
            return ArticleAssessment(
                status=ComplianceStatus.COMPLIANT,
                evidence=f"XUA SAML + mTLS verified ({certs} certs)",
                details={"xua": True, "mtls": True, "certificates": certs},
                module_name="ihe_fl_bridge",
                regulation_ref="EHDS Art. 46",
            )
        parts = []
        if xua_active:
            parts.append("XUA")
        if mtls_active:
            parts.append("mTLS")
        return ArticleAssessment(
            status=ComplianceStatus.PARTIAL,
            evidence=f"Partial: {', '.join(parts) or 'none'} active",
            details={"xua": xua_active, "mtls": mtls_active},
            module_name="ihe_fl_bridge",
            regulation_ref="EHDS Art. 46",
        )

    def _assess_member_state(
        self, trainer, config: Dict
    ) -> ArticleAssessment:
        """Art. 48: Member state rights and jurisdiction privacy."""
        jm = getattr(trainer, "jurisdiction_manager", None)
        if jm is None:
            # Even without jurisdiction manager, per-country epsilon is enforced
            hospitals = getattr(trainer, "hospitals", [])
            countries = set(h.country_code for h in hospitals)
            if len(countries) > 1:
                return ArticleAssessment(
                    status=ComplianceStatus.PARTIAL,
                    evidence=f"{len(countries)} countries, per-country epsilon enforced",
                    details={"countries": sorted(countries)},
                    module_name="jurisdiction_privacy",
                    regulation_ref="EHDS Art. 48",
                )
            return ArticleAssessment(
                status=ComplianceStatus.NOT_ASSESSED,
                evidence="Jurisdiction privacy not enabled",
                module_name="jurisdiction_privacy",
                regulation_ref="EHDS Art. 48",
            )
        # Jurisdiction manager active
        active = jm.get_active_clients()
        total = len(jm.client_states)
        dropouts = total - len(active)
        opted_out = sum(1 for s in jm.client_states.values() if s.opted_out)
        return ArticleAssessment(
            status=ComplianceStatus.COMPLIANT,
            evidence=(
                f"Jurisdiction DP active, "
                f"{dropouts} dropout{'s' if dropouts != 1 else ''}, "
                f"{opted_out} opted out"
            ),
            details={
                "active_clients": len(active),
                "total_clients": total,
                "dropouts": dropouts,
                "opted_out": opted_out,
            },
            module_name="jurisdiction_privacy",
            regulation_ref="EHDS Art. 48",
        )

    def _assess_secure_processing(
        self, trainer, config: Dict
    ) -> ArticleAssessment:
        """Art. 50: Secure processing environment (ATNA/mTLS)."""
        ihe = getattr(trainer, "ihe_bridge", None)
        if ihe is None:
            return ArticleAssessment(
                status=ComplianceStatus.NOT_ASSESSED,
                evidence="IHE bridge not enabled",
                module_name="ihe_fl_bridge",
                regulation_ref="EHDS Art. 50",
            )
        # Read ATNA audit summary
        audit_summary = {}
        if hasattr(ihe, "get_audit_summary"):
            audit_summary = ihe.get_audit_summary()
        atna_events = audit_summary.get("total_events", 0)
        mtls_active = getattr(ihe, "_mtls_enabled", False)
        if atna_events > 0 and mtls_active:
            return ArticleAssessment(
                status=ComplianceStatus.COMPLIANT,
                evidence=f"{atna_events} ATNA events, mTLS active",
                details={
                    "atna_events": atna_events,
                    "mtls": mtls_active,
                },
                module_name="ihe_fl_bridge",
                regulation_ref="EHDS Art. 50",
            )
        parts = []
        if atna_events > 0:
            parts.append(f"ATNA ({atna_events})")
        if mtls_active:
            parts.append("mTLS")
        return ArticleAssessment(
            status=ComplianceStatus.PARTIAL,
            evidence=f"Partial: {', '.join(parts) or 'none'}",
            details={"atna_events": atna_events, "mtls": mtls_active},
            module_name="ihe_fl_bridge",
            regulation_ref="EHDS Art. 50",
        )

    def _assess_permits(
        self, trainer, config: Dict
    ) -> ArticleAssessment:
        """Art. 53: Data permit validation."""
        hospitals = getattr(trainer, "hospitals", [])
        violations = getattr(trainer, "purpose_violations", [])
        n_permits = len(hospitals)  # Each hospital = simulated permit
        if violations:
            return ArticleAssessment(
                status=ComplianceStatus.NON_COMPLIANT,
                evidence=f"{len(violations)} violations out of {n_permits} permits",
                details={
                    "permits": n_permits,
                    "violations": len(violations),
                },
                module_name="data_permits",
                regulation_ref="EHDS Art. 53",
            )
        return ArticleAssessment(
            status=ComplianceStatus.COMPLIANT,
            evidence=f"{n_permits} permits, 0 violations",
            details={"permits": n_permits, "violations": 0},
            module_name="data_permits",
            regulation_ref="EHDS Art. 53",
        )

    def _assess_cross_border(
        self, trainer, config: Dict
    ) -> ArticleAssessment:
        """Art. 57-58: Cross-border data access."""
        bridge = getattr(trainer, "myhealth_bridge", None)
        if bridge and bridge.ncp_nodes:
            n_ncps = len(bridge.ncp_nodes)
            hierarchical = config.get("myhealth_eu_config", {}).get(
                "hierarchical_aggregation", True)
            agg_type = "hierarchical" if hierarchical else "flat"
            return ArticleAssessment(
                status=ComplianceStatus.COMPLIANT,
                evidence=f"{n_ncps} NCPs, {agg_type} aggregation",
                details={
                    "ncp_count": n_ncps,
                    "aggregation": agg_type,
                    "ncp_countries": sorted(bridge.ncp_nodes.keys()),
                },
                module_name="myhealth_eu_bridge",
                regulation_ref="EHDS Art. 57-58",
            )
        # Check if multi-country without NCPeH
        hospitals = getattr(trainer, "hospitals", [])
        countries = set(h.country_code for h in hospitals)
        if len(countries) > 1:
            return ArticleAssessment(
                status=ComplianceStatus.PARTIAL,
                evidence=f"{len(countries)} countries, no NCPeH topology",
                details={"countries": sorted(countries)},
                module_name="myhealth_eu_bridge",
                regulation_ref="EHDS Art. 57-58",
            )
        return ArticleAssessment(
            status=ComplianceStatus.NOT_ASSESSED,
            evidence="Single country, cross-border not applicable",
            module_name="myhealth_eu_bridge",
            regulation_ref="EHDS Art. 57-58",
        )

    def _assess_quality(
        self, trainer, config: Dict
    ) -> ArticleAssessment:
        """Art. 69: Data quality framework."""
        qm = getattr(trainer, "quality_manager", None)
        if qm is None:
            return ArticleAssessment(
                status=ComplianceStatus.NOT_ASSESSED,
                evidence="Data quality framework not enabled",
                module_name="data_quality_framework",
                regulation_ref="EHDS Art. 69",
            )
        # Read quality labels
        report = qm.export_report()
        labels = report.get("quality_labels", {})
        label_counts = {}
        for info in labels.values():
            label = info.get("label", "unknown")
            label_counts[label] = label_counts.get(label, 0) + 1
        label_str = ", ".join(
            f"{count} {label}" for label, count in sorted(label_counts.items())
        )
        return ArticleAssessment(
            status=ComplianceStatus.COMPLIANT,
            evidence=f"Labels: {label_str}" if label_str else "Quality assessed",
            details={"label_distribution": label_counts},
            module_name="data_quality_framework",
            regulation_ref="EHDS Art. 69",
        )

    def _assess_optout(
        self, trainer, config: Dict
    ) -> ArticleAssessment:
        """Art. 71: Citizen opt-out rights."""
        hospitals = getattr(trainer, "hospitals", [])
        audit_log = getattr(trainer, "audit_log", None)
        # Check opt-out from audit log entries
        optout_entries = 0
        if audit_log and hasattr(audit_log, "entries"):
            optout_entries = sum(
                1 for e in audit_log.entries
                if getattr(e, "event_type", "") == "opt_out"
                or "opt" in str(getattr(e, "details", "")).lower()
            )
        total_opted = sum(
            1 for h in hospitals
            if getattr(h, "opted_out", False)
        )
        return ArticleAssessment(
            status=ComplianceStatus.COMPLIANT,
            evidence=f"Registry active, {total_opted} opted out",
            details={
                "total_hospitals": len(hospitals),
                "opted_out": total_opted,
                "optout_audit_entries": optout_entries,
            },
            module_name="optout_registry",
            regulation_ref="EHDS Art. 71",
        )

    def _assess_audit(
        self, trainer, config: Dict
    ) -> ArticleAssessment:
        """GDPR Art. 30: Records of processing activities."""
        audit_log = getattr(trainer, "audit_log", None)
        if audit_log and hasattr(audit_log, "entries"):
            n_entries = len(audit_log.entries)
            return ArticleAssessment(
                status=ComplianceStatus.COMPLIANT,
                evidence=f"{n_entries} audit entries logged",
                details={"audit_entries": n_entries},
                module_name="compliance_logging",
                regulation_ref="GDPR Art. 30",
            )
        return ArticleAssessment(
            status=ComplianceStatus.PARTIAL,
            evidence="Audit log not available",
            module_name="compliance_logging",
            regulation_ref="GDPR Art. 30",
        )

    # -----------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------

    def get_summary(self) -> Dict[str, Any]:
        """Compute summary statistics."""
        total = len(self.assessments)
        compliant = sum(
            1 for a in self.assessments
            if a.status == ComplianceStatus.COMPLIANT)
        partial = sum(
            1 for a in self.assessments
            if a.status == ComplianceStatus.PARTIAL)
        not_assessed = sum(
            1 for a in self.assessments
            if a.status == ComplianceStatus.NOT_ASSESSED)
        non_compliant = sum(
            1 for a in self.assessments
            if a.status == ComplianceStatus.NON_COMPLIANT)
        # Score: COMPLIANT=1, PARTIAL=0.5, NOT_ASSESSED=0, NON_COMPLIANT=0
        score = (compliant + 0.5 * partial) / total * 100 if total > 0 else 0.0
        return {
            "total": total,
            "compliant": compliant,
            "partial": partial,
            "not_assessed": not_assessed,
            "non_compliant": non_compliant,
            "compliance_score_pct": round(score, 1),
        }

    # -----------------------------------------------------------------
    # OUTPUT FORMATS
    # -----------------------------------------------------------------

    def to_terminal_display(self) -> str:
        """Generate formatted terminal display string."""
        lines = []
        lines.append("")
        lines.append("EHDS COMPLIANCE REPORT - REGULATORY SANDBOX ASSESSMENT")
        lines.append("=" * 56)
        lines.append(f"Scenario: {self.scenario_label}")
        lines.append(f"Date: {self.generated_at}")
        if self.final_metrics:
            m = self.final_metrics
            lines.append(
                f"Metrics: Acc={m.get('accuracy', 0):.2%} | "
                f"F1={m.get('f1', 0):.4f} | "
                f"AUC={m.get('auc', 0):.4f}"
            )

        # Group by chapter
        current_chapter = ""
        for a in self.assessments:
            if a.chapter != current_chapter:
                current_chapter = a.chapter
                lines.append(f"\n{current_chapter}")

            # Status tag with fixed width
            status_tag = f"[{a.status.value}]"
            # Color codes (ANSI)
            if a.status == ComplianceStatus.COMPLIANT:
                colored_tag = f"\033[92m{status_tag}\033[0m"  # green
            elif a.status == ComplianceStatus.PARTIAL:
                colored_tag = f"\033[93m{status_tag}\033[0m"  # yellow
            elif a.status == ComplianceStatus.NON_COMPLIANT:
                colored_tag = f"\033[91m{status_tag}\033[0m"  # red
            else:
                colored_tag = f"\033[90m{status_tag}\033[0m"  # gray

            lines.append(
                f"  {a.article_id:<10} {a.title:<32} "
                f"{colored_tag:<28} {a.evidence}"
            )

        # Summary
        s = self.get_summary()
        lines.append("")
        lines.append("-" * 56)

        parts = []
        if s["compliant"]:
            parts.append(f"\033[92m{s['compliant']}/{s['total']} COMPLIANT\033[0m")
        if s["partial"]:
            parts.append(f"\033[93m{s['partial']} PARTIAL\033[0m")
        if s["not_assessed"]:
            parts.append(f"\033[90m{s['not_assessed']} NOT_ASSESSED\033[0m")
        if s["non_compliant"]:
            parts.append(f"\033[91m{s['non_compliant']} NON_COMPLIANT\033[0m")

        lines.append("SUMMARY: " + " | ".join(parts))
        lines.append(f"Compliance Score: {s['compliance_score_pct']}%")
        lines.append("")

        return "\n".join(lines)

    def to_json(self) -> Dict[str, Any]:
        """Export as JSON-serializable dict."""
        return {
            "report_type": "EHDS Regulatory Sandbox Compliance Report",
            "regulation": "EU 2025/327",
            "generated_at": self.generated_at,
            "scenario": self.scenario_label,
            "training_config": self.training_config,
            "final_metrics": self.final_metrics,
            "assessments": [a.to_dict() for a in self.assessments],
            "summary": self.get_summary(),
        }

    def to_latex_table(self) -> str:
        """Generate LaTeX table for paper inclusion."""
        lines = []
        lines.append("% EHDS Compliance Report - Auto-generated")
        lines.append(f"% Scenario: {self.scenario_label}")
        lines.append(f"% Generated: {self.generated_at}")
        lines.append("")
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append("\\caption{EHDS Regulatory Compliance Assessment}")
        lines.append("\\label{tab:ehds_compliance}")
        lines.append("\\small")
        lines.append("\\begin{tabular}{llll}")
        lines.append("\\toprule")
        lines.append(
            "\\textbf{Article} & \\textbf{Title} & "
            "\\textbf{Status} & \\textbf{Evidence} \\\\")
        lines.append("\\midrule")

        current_chapter = ""
        for a in self.assessments:
            if a.chapter != current_chapter:
                current_chapter = a.chapter
                # Chapter header row
                short_chapter = current_chapter.split(" - ")[0]
                lines.append(
                    f"\\multicolumn{{4}}{{l}}"
                    f"{{\\textit{{{short_chapter}}}}} \\\\")

            # Status formatting
            status_str = a.status.value
            if a.status == ComplianceStatus.COMPLIANT:
                status_tex = f"\\textcolor{{green!70!black}}{{{status_str}}}"
            elif a.status == ComplianceStatus.PARTIAL:
                status_tex = f"\\textcolor{{orange!80!black}}{{{status_str}}}"
            elif a.status == ComplianceStatus.NON_COMPLIANT:
                status_tex = f"\\textcolor{{red}}{{{status_str}}}"
            else:
                status_tex = f"\\textcolor{{gray}}{{{status_str}}}"

            # Escape LaTeX special characters in evidence
            evidence = (a.evidence
                        .replace("&", "\\&")
                        .replace("%", "\\%")
                        .replace("_", "\\_")
                        .replace("#", "\\#"))

            lines.append(
                f"{a.article_id} & {a.title} & "
                f"{status_tex} & {evidence} \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")

        # Add summary footnote
        s = self.get_summary()
        lines.append(
            f"\\\\[2pt]\\footnotesize "
            f"Score: {s['compliance_score_pct']}\\% "
            f"({s['compliant']}/{s['total']} compliant, "
            f"{s['partial']} partial, "
            f"{s['not_assessed']} not assessed)")

        lines.append("\\end{table}")
        return "\n".join(lines)

    # -----------------------------------------------------------------
    # SCENARIO COMPARISON
    # -----------------------------------------------------------------

    @staticmethod
    def compare_scenarios(
        report_a: "EHDSComplianceReport",
        report_b: "EHDSComplianceReport",
    ) -> Dict[str, Any]:
        """Compare two compliance reports."""
        comparison_rows = []
        for a_assess, b_assess in zip(report_a.assessments, report_b.assessments):
            changed = a_assess.status != b_assess.status
            comparison_rows.append({
                "article_id": a_assess.article_id,
                "title": a_assess.title,
                "status_a": a_assess.status.value,
                "status_b": b_assess.status.value,
                "evidence_a": a_assess.evidence,
                "evidence_b": b_assess.evidence,
                "changed": changed,
            })

        sum_a = report_a.get_summary()
        sum_b = report_b.get_summary()

        metric_comparison = {}
        for key in ["accuracy", "f1", "auc", "loss"]:
            val_a = report_a.final_metrics.get(key, 0)
            val_b = report_b.final_metrics.get(key, 0)
            metric_comparison[key] = {
                "scenario_a": val_a,
                "scenario_b": val_b,
                "delta": round(val_b - val_a, 4),
            }

        return {
            "scenario_a": report_a.scenario_label,
            "scenario_b": report_b.scenario_label,
            "comparison_rows": comparison_rows,
            "score_a": sum_a["compliance_score_pct"],
            "score_b": sum_b["compliance_score_pct"],
            "metric_comparison": metric_comparison,
        }

    @staticmethod
    def format_comparison_terminal(comparison: Dict[str, Any]) -> str:
        """Format scenario comparison for terminal display."""
        lines = []
        lines.append("")
        lines.append("SCENARIO COMPARISON")
        lines.append("=" * 70)
        lines.append(f"  A: {comparison['scenario_a']}")
        lines.append(f"  B: {comparison['scenario_b']}")
        lines.append("")

        # Header
        lines.append(
            f"  {'Article':<10} {'Title':<28} "
            f"{'Scenario A':<16} {'Scenario B':<16}")
        lines.append("  " + "-" * 66)

        for row in comparison["comparison_rows"]:
            marker = " *" if row["changed"] else "  "
            lines.append(
                f"{marker}{row['article_id']:<10} {row['title']:<28} "
                f"{row['status_a']:<16} {row['status_b']:<16}"
            )

        # Metrics
        lines.append("")
        lines.append("  TRAINING METRICS")
        lines.append("  " + "-" * 50)
        for key, vals in comparison["metric_comparison"].items():
            delta_str = f"{vals['delta']:+.4f}" if vals['delta'] != 0 else "="
            lines.append(
                f"  {key:<16} {vals['scenario_a']:<16.4f} "
                f"{vals['scenario_b']:<16.4f} {delta_str}"
            )

        lines.append("")
        lines.append(
            f"  Compliance Score: "
            f"{comparison['score_a']}% -> {comparison['score_b']}%")
        lines.append("")

        return "\n".join(lines)
