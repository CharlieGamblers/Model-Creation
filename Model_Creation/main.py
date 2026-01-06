from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    cur = Path(__file__).resolve()
    for p in cur.parents:
        if p.name == "Bull_Model":
            return p
    raise FileNotFoundError("folder 'Bull_Model' not found up-tree")


def _py() -> str:
    return sys.executable or "python"


def run_step(description: str, rel_path: Path) -> None:
    root = _repo_root()
    script_path = root / rel_path
    print(f"\n[run] {description} → {script_path}", flush=True)
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    # Run from repo root using a short relative path to avoid Windows long-path issues
    completed = subprocess.run([_py(), str(rel_path)], cwd=str(root), check=True)
    if completed.returncode == 0:
        print(f"[ok] {description}", flush=True)


def main() -> None:
    root = _repo_root()

    # 1) Ingest raw data
    run_step("Ingest data from API", Path("Scripts") / "Ingest" / "fetch_data.py")

    # 2) Clean / normalize
    run_step("Clean and normalize raw data", Path("Scripts") / "Clean" / "clean_data.py")

    # 3) Build engineered datasets
    run_step(
        "Build engineered datasets (rides_data_final.csv, final_data.csv)",
        Path("Scripts") / "Build" / "build_dataset.py",
    )

    # 4) Train model
    run_step("Train XGBoost model and export artifacts", Path("Scripts") / "Train" / "model_train.py")

    print("\n[✓] Model creation pipeline complete.", flush=True)


if __name__ == "__main__":
    main()

