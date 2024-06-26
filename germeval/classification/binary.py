from typing import Union

import evaluate
import numpy as np
import torch
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
)

from germeval.classification.utils import (
    get_training_args,
)


def train_binary_classifier(
    dataset: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset],
    params: dict,
) -> None:
    training_args = get_training_args(params)
    input_col = params.get("input_col", "article")

    model_name = params["model_name"]
    label_col = params.get("label_col", None)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess_fn(examples: dict) -> dict:
        return tokenizer(examples[input_col], truncation=True, padding=True)

    def add_label_col(examples: dict, label_col: str) -> dict:
        examples["label"] = np.array(examples[label_col]).astype(int).tolist()
        return examples

    def compute_metrics(eval_pred: tuple) -> dict:
        roc_auc = evaluate.load("roc_auc")
        metrics = evaluate.combine(["accuracy", "recall", "f1", "precision"])
        predictions, labels = eval_pred

        prediction_softmax_scores = torch.softmax(torch.tensor(predictions), 1)[
            :,
            1,
        ].numpy()
        prediction_classes = np.argmax(predictions, axis=1)
        output = metrics.compute(
            predictions=prediction_classes,
            references=labels,
        )
        output.update(
            roc_auc.compute(
                predictions=predictions,
                references=labels,
                prediction_scores=prediction_softmax_scores,
            )
        )

        return output

    tokenized_dataset = dataset.map(preprocess_fn, batched=True)
    if not all("label" in split_cols for split_cols in tokenized_dataset.values()):
        tokenized_dataset = tokenized_dataset.map(
            add_label_col, fn_kwargs={"label_col": label_col}
        )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    )

    model.config.label2id = {
        label: idx for idx, label in enumerate(params["model_labels"])
    }
    model.config.id2label = {
        str(idx): label for idx, label in enumerate(params["model_labels"])
    }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
