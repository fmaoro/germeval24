import os

from transformers import (
    TrainingArguments,
)


def get_training_args(params: dict) -> TrainingArguments:
    training_params = params["training_args"]
    args = dict(training_params.items())
    if "learning_rate" in args:
        args["learning_rate"] = float(args["learning_rate"])
    if "output_dir" in args:
        args["output_dir"] = os.path.join(args["output_dir"])

    training_args = TrainingArguments(**args)
    return training_args
