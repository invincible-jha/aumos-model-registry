"""Intelligent model decommissioning for zombie AI model lifecycle management.

This package implements automated end-of-life management for AI models.
It scores models against five weighted signals to identify candidates for
decommissioning and orchestrates a safe state-machine-driven workflow that
drains traffic gradually before archiving and deleting the model.

Modules:
    signal_collector — DecommissionSignalCollector: collects and scores 5 signals
    workflow_manager — DecommissionWorkflowManager: state machine + traffic drain
"""

from aumos_model_registry.decommission.signal_collector import DecommissionSignalCollector
from aumos_model_registry.decommission.workflow_manager import DecommissionWorkflowManager

__all__ = ["DecommissionSignalCollector", "DecommissionWorkflowManager"]
