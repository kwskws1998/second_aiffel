import inspect

from transformers import TrainingArguments


def build_training_arguments(**kwargs):
    signature = inspect.signature(TrainingArguments.__init__)
    supported = set(signature.parameters)
    filtered = {key: value for key, value in kwargs.items() if key in supported}

    if "evaluation_strategy" in kwargs and "evaluation_strategy" not in supported and "eval_strategy" in supported:
        filtered["eval_strategy"] = kwargs["evaluation_strategy"]

    dropped = sorted(set(kwargs) - set(filtered))
    if dropped:
        print("[TrainingArguments] dropped unsupported args:", ", ".join(dropped))

    return TrainingArguments(**filtered)
