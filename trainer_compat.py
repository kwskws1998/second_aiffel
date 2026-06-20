import inspect

from transformers import Trainer


def build_trainer(trainer_cls, *args, **kwargs):
    signature = inspect.signature(Trainer.__init__)
    supported = set(signature.parameters)

    filtered = {}
    for key, value in kwargs.items():
        if key in supported:
            filtered[key] = value

    if "tokenizer" in kwargs and "tokenizer" not in supported and "processing_class" in supported:
        filtered["processing_class"] = kwargs["tokenizer"]

    dropped = sorted(set(kwargs) - set(filtered))
    if "processing_class" in filtered and "tokenizer" in dropped:
        dropped.remove("tokenizer")
    if dropped:
        print("[Trainer] dropped unsupported args:", ", ".join(dropped))

    return trainer_cls(*args, **filtered)
