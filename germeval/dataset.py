import json
import math
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Literal

from datasets import DatasetDict, load_dataset
from pydantic import BaseModel, model_validator


class Sample(BaseModel):
    id: str
    text: str


class Label(str, Enum):
    KEIN = "0-Kein"
    GERING = "1-Gering"
    VORHANDEN = "2-Vorhanden"
    STARK = "3-Stark"
    EXTREM = "4-Extrem"


class Annotation(BaseModel):
    user: str
    label: Label


class TrainSample(Sample):
    annotations: list[Annotation]


class TestSample(Sample):
    annotators: list[str]


class SampleLabels(BaseModel):
    bin_maj: int  # bin_maj: predict 1 if a majority of annotators assigned a label other than 0-Kein, predict 0 if a majority of annotators assigned a label 0-Kein. If there was no majority, then both the label 1 and 0 will count as correct in the evaluation.
    bin_one: int  # bin_one: predict 1 if at least one annotator assigned a label other than 0-Kein, 0 otherwise
    bin_all: int  # bin_all: predict 1 if all annotators assigned labels other than 0-Kein, 0 otherwise
    multi_maj: Label  # multi_maj: predict the majority label if there is one, if there is no majority label, any of the labels assigned is counted as a correct prediction for evaluation
    disagree_bin: int  # disagree_bin: predict 1 if there is disagreement between annotators on 0-Kein versus all other labels and 0 otherwise


class LabeledSample(BaseModel):
    sample: TrainSample
    labels: SampleLabels


class TargetSample(SampleLabels):
    id: str


class DistLabel(BaseModel):
    label: Label
    score: float


class ST2TargetSample(BaseModel):
    id: str
    dist_bin_0: float
    dist_bin_1: float
    dist_multi_0: float
    dist_multi_1: float
    dist_multi_2: float
    dist_multi_3: float
    dist_multi_4: float

    @model_validator(mode="after")
    def check_bin_sum(self) -> "ST2TargetSample":
        bin_sum = math.fsum([self.dist_bin_0 + self.dist_bin_1])
        if bin_sum != 1:
            raise ValueError(
                f"Sum of dist_bin_0 ({self.dist_bin_0}) and dist_bin_1 ({self.dist_bin_1}) must add up to 1 ({bin_sum})"
            )
        return self

    @model_validator(mode="after")
    def check_multi_sum(self) -> "ST2TargetSample":
        multi_sum = math.fsum(
            [
                self.dist_multi_0,
                self.dist_multi_1,
                self.dist_multi_2,
                self.dist_multi_3,
                self.dist_multi_4,
            ]
        )
        if multi_sum != 1:
            raise ValueError(f"Sum of dist_multi must add up to 1 (is {multi_sum})")
        return self


def read_jsonl(path: Path) -> list:
    lines = []
    with open(path, "r") as input_file:
        for line in input_file.readlines():
            lines.append(json.loads(line))

    return lines


def get_bin_maj(sample: TrainSample) -> bool:
    labels = [annotation.label for annotation in sample.annotations]
    num_items: dict[Label, int] = defaultdict(int)
    for label in labels:
        num_items[label] += 1
    if num_items[Label.KEIN] == max(list(num_items.values())):
        return False
    else:
        return True


def get_bin_one(sample: TrainSample) -> bool:
    labels = [annotation.label for annotation in sample.annotations]
    if any(label != Label.KEIN for label in labels):
        return True
    else:
        return False


def get_bin_all(sample: TrainSample) -> bool:
    labels = [annotation.label for annotation in sample.annotations]
    if all(label != Label.KEIN for label in labels):
        return True
    else:
        return False


def get_multi_maj(sample: TrainSample) -> Label:
    labels = [annotation.label for annotation in sample.annotations]
    num_items: dict[Label, int] = defaultdict(int)
    for label in labels:
        num_items[label] += 1

    return sorted(num_items.items(), key=lambda x: x[1], reverse=True)[0][0]


def get_disagree_bin(sample: TrainSample) -> bool:
    labels = [annotation.label for annotation in sample.annotations]
    if any(label == Label.KEIN for label in labels) and not all(
        label == Label.KEIN for label in labels
    ):
        return True
    else:
        return False


def get_multi_all(maj: bool, all: bool) -> Literal["maj", "all", "no_maj"]:  # noqa: A002
    if all is True:
        return "all"
    elif maj is True:
        return "maj"
    else:
        return "no_maj"


def add_labels(example: dict) -> dict:
    sample = TrainSample(**example)
    labels = SampleLabels(
        bin_maj=get_bin_maj(sample=sample),
        bin_one=get_bin_one(sample=sample),
        bin_all=get_bin_all(sample=sample),
        multi_maj=get_multi_maj(sample=sample),
        disagree_bin=get_disagree_bin(sample=sample),
    )
    example["bin_maj"] = labels.bin_maj
    example["bin_one"] = labels.bin_one
    example["bin_all"] = labels.bin_all
    example["multi_maj"] = labels.multi_maj.value
    example["disagree_bin"] = labels.disagree_bin
    example["multi_all"] = get_multi_all(
        maj=bool(labels.bin_maj), all=bool(labels.bin_all)
    )
    return example


def create_germeval24_dataset(train_path: Path) -> DatasetDict:
    ds = load_dataset("json", data_files=str(train_path), split="train")
    # splitted = ds.train_test_split(test_size=0.2)
    splitted = DatasetDict({"train": ds, "validation": ds})
    # splitted = ds
    # splitted["validation"] = splitted["train"]
    # del splitted["test"]

    processed_ds = splitted.map(add_labels)
    return processed_ds


def filter_bin_one_is_true(example: dict) -> bool:
    if example["bin_one"] == 1:
        return True
    else:
        return False


def filter_at_least_maj(example: dict) -> bool:
    if example["bin_maj"] == 1 or example["bin_all"] == 1:
        return True
    else:
        return False


def filter_bin_all_is_false(example: dict) -> bool:
    if example["bin_all"] == 0:
        return True
    else:
        return False
