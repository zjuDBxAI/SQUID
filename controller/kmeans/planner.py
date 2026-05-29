from __future__ import annotations

from .hybrid_planner import HybridACLKMeansPlanner


class TenantKMeansPlanner(HybridACLKMeansPlanner):
    """Backward-compatible entry point for the hybrid ACL planner."""
