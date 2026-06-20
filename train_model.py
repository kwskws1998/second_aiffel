import argparse
import json
import os
import socket
from datetime import datetime
from signal import signal

from data_loader import MyDataset
from fold1 import training_fold1
from fold2 import training_fold2
from utils import create_prediction_tables, handle_signal


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

MODEL_CHOICES = ["distilbert", "xlmroberta-base", "xlmroberta-large"]
LOSS_CHOICES = ["mse", "ccc", "robust", "mse+ccc", "robust+ccc"]
MODEL_TO_CHECKPOINT = {
    "distilbert": "distilbert-base-multilingual-cased",
    "xlmroberta-base": "xlm-roberta-base",
    "xlmroberta-large": "xlm-roberta-large",
}


def _parse_features_used(raw_value):
    try:
        parsed = [int(x.strip()) for x in str(raw_value).split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("features-used must be comma-separated integers.") from exc
    if len(parsed) != 5:
        raise argparse.ArgumentTypeError("features-used must have 5 values: nFix,FFD,GPT,TRT,fixProp.")
    if any(value not in (0, 1) for value in parsed):
        raise argparse.ArgumentTypeError("features-used values must be 0 or 1.")
    if sum(parsed) == 0:
        raise argparse.ArgumentTypeError("features-used must enable at least one feature.")
    return parsed


def _parse_fp_dropout(raw_value):
    try:
        parsed = [float(x.strip()) for x in str(raw_value).split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("fp-dropout must be comma-separated floats.") from exc
    if len(parsed) != 2:
        raise argparse.ArgumentTypeError("fp-dropout must have exactly two values.")
    return parsed


def _build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODEL_CHOICES)
    parser.add_argument("loss", choices=LOSS_CHOICES)
    parser.add_argument("--checkpoint-override", default=None)
    parser.add_argument("--gaze-fusion", choices=["none", "add", "concat"], default="none")
    parser.add_argument("--use-gaze-add", action="store_true")
    parser.add_argument("--use-gaze-concat", action="store_true")
    parser.add_argument("--et2-checkpoint", default=None)
    parser.add_argument("--features-used", type=_parse_features_used, default=[1, 1, 1, 1, 1])
    parser.add_argument("--fp-dropout", type=_parse_fp_dropout, default=[0.0, 0.3])
    parser.add_argument("--gaze-add-scale", type=float, default=0.05)
    parser.add_argument("--train-gaze-add-scale", action="store_true")
    parser.add_argument("--no-load-fixation-model", action="store_true")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--maxlen", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batch-size-distil", type=int, default=None)
    parser.add_argument("--batch-size-xlmrb", dest="batch_size_xlmrB", type=int, default=None)
    parser.add_argument("--batch-size-xlmrl", dest="batch_size_xlmrL", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=6e-6)
    parser.add_argument("--train-epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-strategy", choices=["epoch", "no"], default="epoch")
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--save-final-model", dest="save_final_model", action="store_true")
    parser.add_argument("--no-save-final-model", dest="save_final_model", action="store_false")
    parser.add_argument("--load-best-model-at-end", dest="load_best_model_at_end", action="store_true")
    parser.add_argument("--no-load-best-model-at-end", dest="load_best_model_at_end", action="store_false")
    parser.set_defaults(save_final_model=True)
    parser.set_defaults(load_best_model_at_end=True)
    return parser


def _resolve_gaze_fusion(args):
    legacy_flags = int(args.use_gaze_add) + int(args.use_gaze_concat)
    if legacy_flags > 1:
        raise ValueError("--use-gaze-add and --use-gaze-concat are mutually exclusive.")
    if args.gaze_fusion != "none" and legacy_flags:
        raise ValueError("Use either --gaze-fusion or legacy --use-gaze-add/--use-gaze-concat.")
    if args.use_gaze_add:
        return "add"
    if args.use_gaze_concat:
        return "concat"
    return args.gaze_fusion


def _validate_args(args):
    if args.maxlen <= 0:
        raise ValueError("--maxlen must be > 0.")
    if args.train_epochs <= 0:
        raise ValueError("--train-epochs must be > 0.")
    if args.max_steps < -1 or args.max_steps == 0:
        raise ValueError("--max-steps must be -1 or > 0.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0.")
    if args.gaze_add_scale < 0:
        raise ValueError("--gaze-add-scale must be >= 0.")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be > 0.")
    if args.save_total_limit <= 0:
        raise ValueError("--save-total-limit must be > 0.")
    if args.save_strategy == "no" and args.load_best_model_at_end:
        args.load_best_model_at_end = False
        print("[train_model] save_strategy=no, so load_best_model_at_end was set to False.")


def _create_run_dir():
    timestamp = datetime.now().strftime("%b-%d_%H-%M-%S")
    host_name = os.environ.get("COMPUTERNAME") or os.environ.get("HOST") or socket.gethostname()
    preds_dir = f"Preds/{timestamp}_{host_name}"
    os.makedirs(preds_dir, exist_ok=False)
    return timestamp, preds_dir


def _require_dataset_files(data_dir):
    expected = [
        os.path.join(data_dir, "full_dataset_fold1.csv"),
        os.path.join(data_dir, "full_dataset_fold2.csv"),
    ]
    missing = [path for path in expected if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            "Missing dataset file(s):\n"
            + "\n".join(f"  - {path}" for path in missing)
            + "\nBuild no-IEMOCAP data with:\n"
            + "  python setup_no_iemocap_data.py --data-dir data --output-dir data_no_iemocap --seed 42"
        )


def main():
    signal(2, handle_signal)
    parser = _build_parser()
    args = parser.parse_args()
    try:
        gaze_fusion = _resolve_gaze_fusion(args)
        _validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    checkpoint = args.checkpoint_override or MODEL_TO_CHECKPOINT[args.model]
    timestamp, preds_dir = _create_run_dir()

    batch_size_distil = args.batch_size_distil if args.batch_size_distil is not None else args.batch_size
    batch_size_xlmrB = args.batch_size_xlmrB if args.batch_size_xlmrB is not None else args.batch_size
    batch_size_xlmrL = args.batch_size_xlmrL if args.batch_size_xlmrL is not None else args.batch_size
    params = {
        "batch_size_distil": batch_size_distil,
        "batch_size_xlmrB": batch_size_xlmrB,
        "batch_size_xlmrL": batch_size_xlmrL,
        "lr": args.learning_rate,
        "train_epochs": args.train_epochs,
        "max_steps": args.max_steps,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "optim": args.optim,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "seed": args.seed,
        "save_strategy": args.save_strategy,
        "save_total_limit": args.save_total_limit,
        "save_final_model": args.save_final_model,
        "load_best_model_at_end": args.load_best_model_at_end,
    }
    gaze_config = {
        "gaze_fusion": gaze_fusion,
        "et2_checkpoint_path": args.et2_checkpoint,
        "features_used": args.features_used,
        "fp_dropout": args.fp_dropout,
        "gaze_add_scale": args.gaze_add_scale,
        "train_gaze_add_scale": args.train_gaze_add_scale,
        "load_fixation_model": not args.no_load_fixation_model,
    }
    run_parameters = {
        "model": args.model,
        "loss_function": args.loss,
        "path": preds_dir,
        "checkpoint": checkpoint,
        "checkpoint_override": args.checkpoint_override,
        "data_dir": args.data_dir,
        "maxlen": args.maxlen,
        **params,
        **gaze_config,
    }

    with open(f"{preds_dir}/training_parameters.json", "w") as output_file:
        json.dump(run_parameters, output_file)

    _require_dataset_files(args.data_dir)
    filename_1 = os.path.join(args.data_dir, "full_dataset_fold1.csv")
    filename_2 = os.path.join(args.data_dir, "full_dataset_fold2.csv")
    split_1 = MyDataset(filename=filename_1, checkpoint=checkpoint, maxlen=args.maxlen)
    split_2 = MyDataset(filename=filename_2, checkpoint=checkpoint, maxlen=args.maxlen)
    dataset = [[split_1, split_2], [split_2, split_1]]

    training_fold1(args.model, args.loss, timestamp, params, dataset, preds_dir, checkpoint, gaze_config=gaze_config)
    print("\n\n\n------------ NOW ON FOLD 2 -------------- \n\n\n")
    training_fold2(args.model, args.loss, timestamp, params, dataset, preds_dir, checkpoint, gaze_config=gaze_config)
    create_prediction_tables(preds_dir, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
