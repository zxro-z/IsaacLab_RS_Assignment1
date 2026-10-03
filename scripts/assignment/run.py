"""Run native Isaac Lab scripts using this checkout, not another editable install.

Use with isaaclab.sh -p. ISAAC_SIM_SITE_PACKAGES optionally supplies an existing
Isaac Sim installation after the active Python environment's own dependencies.
This helper changes import resolution only; it does not modify any task config.
"""

import os
from pathlib import Path
import runpy
import sys


def main():
    root = Path(__file__).resolve().parents[2]
    if len(sys.argv) < 2:
        raise SystemExit("Usage: run.py <repository script> [script arguments ...]")
    target = (root / sys.argv.pop(1)).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise SystemExit("The target must be an existing script inside this repository.")
    sim_site = os.environ.get("ISAAC_SIM_SITE_PACKAGES")
    if sim_site:
        if not Path(sim_site).is_dir():
            raise SystemExit("ISAAC_SIM_SITE_PACKAGES must name an existing directory.")
        sys.path.append(sim_site)
    for package in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl"):
        sys.path.insert(0, str(root / "source" / package))
    sys.path.insert(0, str(target.parent))
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
