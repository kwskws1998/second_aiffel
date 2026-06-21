from models import DistilBertForSequenceClassificationSig, XLMRobertaForSequenceClassificationSig
from gaze_models import (
    GazeAddForSequenceRegression,
    GazeConcatForSequenceRegression,
    GazePooledHeadForSequenceRegression,
    GazeWordConcatForSequenceRegression,
)


def build_model(model_name, checkpoint, gaze_config=None, tokenizer=None):
    gaze_config = gaze_config or {}
    gaze_fusion = gaze_config.get("gaze_fusion", "none")

    if gaze_fusion == "add":
        return GazeAddForSequenceRegression(
            model_name=model_name,
            checkpoint=checkpoint,
            tokenizer=tokenizer,
            et2_checkpoint_path=gaze_config.get("et2_checkpoint_path"),
            features_used=gaze_config.get("features_used", [1, 1, 1, 1, 1]),
            fp_dropout=tuple(gaze_config.get("fp_dropout", [0.0, 0.3])),
            gaze_add_scale=gaze_config.get("gaze_add_scale", 0.05),
            train_gaze_add_scale=gaze_config.get("train_gaze_add_scale", False),
            load_fixation_model=gaze_config.get("load_fixation_model", True),
        )

    if gaze_fusion == "concat":
        return GazeConcatForSequenceRegression(
            model_name=model_name,
            checkpoint=checkpoint,
            tokenizer=tokenizer,
            et2_checkpoint_path=gaze_config.get("et2_checkpoint_path"),
            features_used=gaze_config.get("features_used", [1, 1, 1, 1, 1]),
            fp_dropout=tuple(gaze_config.get("fp_dropout", [0.0, 0.3])),
            load_fixation_model=gaze_config.get("load_fixation_model", True),
        )

    if gaze_fusion == "word_concat":
        return GazeWordConcatForSequenceRegression(
            model_name=model_name,
            checkpoint=checkpoint,
            tokenizer=tokenizer,
            et2_checkpoint_path=gaze_config.get("et2_checkpoint_path"),
            features_used=gaze_config.get("features_used", [1, 1, 1, 1, 1]),
            fp_dropout=tuple(gaze_config.get("fp_dropout", [0.0, 0.3])),
            load_fixation_model=gaze_config.get("load_fixation_model", True),
        )

    if gaze_fusion == "pooled_head":
        return GazePooledHeadForSequenceRegression(
            model_name=model_name,
            checkpoint=checkpoint,
            tokenizer=tokenizer,
            et2_checkpoint_path=gaze_config.get("et2_checkpoint_path"),
            features_used=gaze_config.get("features_used", [1, 1, 1, 1, 1]),
            fp_dropout=tuple(gaze_config.get("fp_dropout", [0.0, 0.3])),
            load_fixation_model=gaze_config.get("load_fixation_model", True),
        )

    if model_name == "distilbert":
        return DistilBertForSequenceClassificationSig.from_pretrained(checkpoint, num_labels=2)
    if model_name in ("xlmroberta-base", "xlmroberta-large"):
        return XLMRobertaForSequenceClassificationSig.from_pretrained(checkpoint, num_labels=2)
    raise ValueError(f"Unknown model name: {model_name}")
