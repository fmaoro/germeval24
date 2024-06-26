from typing import List, Union

import evaluate
import numpy as np
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
)

from germeval.classification.utils import get_training_args


def preprocess_fn(examples: dict, tokenizer: AutoTokenizer, input_col: str) -> dict:
    inputs = examples[input_col]
    if all(isinstance(_input, list) for _input in inputs):
        split_into_words = True
    else:
        split_into_words = False

    return tokenizer(
        inputs, truncation=True, padding=True, is_split_into_words=split_into_words
    )


def add_label_col(examples: dict, label_col: str, label2id: dict) -> dict:
    ids = [label2id[label] for label in examples[label_col]]
    examples["label"] = np.array(ids).astype(int).tolist()
    return examples


def train_multiclass_classifier(
    dataset: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset],
    params: dict,
) -> None:
    training_args = get_training_args(params)
    input_col = params.get("input_col", "article")
    label_col = params.get("label_col", None)
    model_labels: List[str] = params.get("model_labels", [])

    model_name = params["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    label2id = {label: idx for idx, label in enumerate(model_labels)}
    id2label = {str(idx): label for idx, label in enumerate(model_labels)}

    dataset = dataset.map(
        preprocess_fn,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "input_col": input_col,
        },
    )
    dataset = dataset.map(
        add_label_col,
        fn_kwargs={"label_col": label_col, "label2id": label2id},
        batched=True,
    )

    def compute_metrics(eval_pred: tuple) -> dict:
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = evaluate.load("accuracy")
        recall = evaluate.load("recall")
        f1 = evaluate.load("f1")
        precision = evaluate.load("precision")
        outputs = {}
        outputs.update(accuracy.compute(predictions=predictions, references=labels))
        outputs.update(
            recall.compute(predictions=predictions, references=labels, average="macro")
        )
        outputs.update(
            f1.compute(predictions=predictions, references=labels, average="macro")
        )
        outputs.update(
            precision.compute(
                predictions=predictions, references=labels, average="macro"
            )
        )

        return outputs

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(model_labels), label2id=label2id, id2label=id2label
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
