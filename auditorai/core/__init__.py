"""
AuditorAI core package.

Exports: AuditorModel, Router, AuditorSystem, audit, run_full_evaluation
"""

from auditorai.core.auditor import AuditorModel
from auditorai.core.router import Router
from auditorai.core.system import AuditorSystem, audit
from auditorai.core.evaluate import run_full_evaluation

__all__ = [
    "AuditorModel",
    "Router",
    "AuditorSystem",
    "audit",
    "run_full_evaluation",
]
