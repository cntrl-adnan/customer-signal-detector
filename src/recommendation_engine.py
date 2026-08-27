"""
Controlled action mapping based on action_map_v1.yaml.
"""
import yaml
import os
from pathlib import Path
from typing import Optional

# Repo root, resolved from this file so the engine works from any CWD.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ACTION_MAP_CONFIG = PROJECT_ROOT / "config" / "action_map_v1.yaml"


class RecommendationEngine:
    def __init__(self, config_path: Optional[str] = None):
        config_path = str(config_path or DEFAULT_ACTION_MAP_CONFIG)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Action map config not found at: {config_path}")

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.actions = self.config.get("actions", {})

    def get_action(self, issue_type: Optional[str], risk_band: str) -> str:
        issue = issue_type or "other"
        issue_rules = self.actions.get(issue, self.actions.get("other", {}))

        if risk_band in ["Critical", "High"] and "high_critical" in issue_rules:
            return issue_rules["high_critical"]
        if risk_band in ["Critical", "High", "Medium"] and "medium_high_critical" in issue_rules:
            return issue_rules["medium_high_critical"]
        if "default" in issue_rules:
            return issue_rules["default"]
        return issue_rules.get("any", "Manual operations review")