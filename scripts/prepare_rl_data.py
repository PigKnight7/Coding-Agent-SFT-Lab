#!/usr/bin/env python3
"""Freeze a read-only RL projection without changing SFT data or original repositories."""
from pathlib import Path
import sys
import json
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cc_agent.rl.data import freeze_tasks

if __name__ == "__main__":
    result = freeze_tasks(ROOT)
    print(json.dumps(result["hidden_assertions"], indent=2))
