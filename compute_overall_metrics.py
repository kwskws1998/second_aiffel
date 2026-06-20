import argparse
import json
import os
from pathlib import Path

from utils import create_prediction_tables


def _resolve_data_dir(run_dir, override_data_dir):
    if override_data_dir:
        return override_data_dir

    params_path = run_dir / "training_parameters.json"
    if params_path.is_file():
        params = json.loads(params_path.read_text())
        data_dir = params.get("data_dir")
        if data_dir:
            return data_dir

    return "data"


def _completed_run(run_dir):
    required = [
        "predictions_fold1.csv",
        "predictions_fold2.csv",
        "training_parameters.json",
    ]
    return all((run_dir / name).is_file() for name in required)


def _recent_runs(preds_dir, count):
    runs = [path for path in preds_dir.iterdir() if path.is_dir()]
    runs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return runs[:count]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="*", help="Preds run directories to process.")
    parser.add_argument("--preds-dir", default="Preds")
    parser.add_argument("--recent", type=int, default=0, help="Process the N most recent Preds runs.")
    parser.add_argument("--data-dir", default=None, help="Override data_dir for all runs.")
    args = parser.parse_args()

    run_dirs = [Path(run) for run in args.runs]
    if args.recent:
        run_dirs.extend(_recent_runs(Path(args.preds_dir), args.recent))
    if not run_dirs:
        parser.error("Pass run directories or --recent N.")

    for run_dir in run_dirs:
        if not _completed_run(run_dir):
            print(f"[skip] incomplete run: {run_dir}")
            continue

        data_dir = _resolve_data_dir(run_dir, args.data_dir)
        create_prediction_tables(str(run_dir), data_dir=data_dir)
        print(f"[ok] wrote metrics: {run_dir}")


if __name__ == "__main__":
    main()
