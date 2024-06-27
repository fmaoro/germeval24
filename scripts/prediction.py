import csv
import logging
import math
import random
from argparse import ArgumentParser
from decimal import Decimal
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from germeval.dataset import (
    DistLabel,
    Label,
    ST2TargetSample,
    TargetSample,
    read_jsonl,
)
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    pipeline,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from typing_extensions import ParamSpec

P = ParamSpec("P")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

DATA_PATH = Path("data/germeval24/germeval-competition-test.jsonl")

M1_PATH = Path("models/germeval24/gbert-large/m1/checkpoint-200")
M2_PATH = Path("models/germeval24/gbert-large/m2/checkpoint-112")
M3_PATH = Path("models/germeval24/gbert-large/m3/checkpoint-276")
M4_PATH = Path("models/germeval24/gbert-large/m4/checkpoint-192")
M5_PATH = Path("models/germeval24/gbert-large/m5/checkpoint-900")
M6_PATH = Path("models/germeval24/gbert-large/m6/checkpoint-168")
M7_PATH = Path("models/germeval24/gbert-large/m7/checkpoint-14")


class TextDataset(Dataset):
    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        super().__init__()

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int) -> str:
        return self.texts[i]


def get_argparser() -> ArgumentParser:
    parser = ArgumentParser("Germeval Prediction Script")
    parser.add_argument("-o", "--output", type=str, help="Output file", required=True)
    parser.add_argument("-a", "--approach", type=int, required=False, default=None)
    return parser


def load_model_and_tokenizer(
    model_path: Path,
) -> tuple[AutoModelForSequenceClassification, PreTrainedTokenizerBase]:
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    if torch.cuda.is_available():
        model.to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def parse_binary_output(outputs: list[SequenceClassifierOutput]) -> list[bool]:
    all_logits = torch.cat([output.logits for output in outputs])
    activated = torch.softmax(all_logits, dim=1)
    argmax = torch.argmax(activated, dim=1)

    labels = []
    for example_pred in argmax:
        labels.append(True if example_pred == 1 else False)
    return labels


def parse_multiclass_output(
    outputs: list[SequenceClassifierOutput], id2label: dict
) -> list[str]:
    all_logits = torch.cat([output.logits for output in outputs])
    activated = torch.softmax(all_logits, dim=1)
    argmax = torch.argmax(activated, dim=1)

    labels = []
    for example_pred in argmax:
        labels.append(id2label[example_pred.item()])
    return labels


def predict_binary_pipe(
    texts: list[str],
    model: AutoModelForSequenceClassification,
    tokenizer: PreTrainedTokenizerBase,
    description: Optional[str],
) -> list[bool]:
    if len(texts) == 0:
        return []
    pipe = pipeline(
        task="text-classification", model=model, tokenizer=tokenizer, device=0
    )
    dataset = TextDataset(texts=texts)
    predictions = list(
        tqdm(pipe(dataset, batch_size=50), total=len(dataset), desc=description)
    )
    label2id = model.config.label2id
    outputs = [True if label2id[pred["label"]] == 1 else False for pred in predictions]
    return outputs


def predict_binary_dist_pipe(
    texts: list[str],
    model: AutoModelForSequenceClassification,
    tokenizer: PreTrainedTokenizerBase,
    description: Optional[str],
) -> list[tuple[bool, float]]:
    if len(texts) == 0:
        return []
    pipe = pipeline(
        task="text-classification", model=model, tokenizer=tokenizer, device=0
    )
    dataset = TextDataset(texts=texts)
    predictions = list(
        tqdm(pipe(dataset, batch_size=50), total=len(dataset), desc=description)
    )
    label2id = model.config.label2id
    outputs = [
        (True, pred["score"])
        if label2id[pred["label"]] == 1
        else (False, pred["score"])
        for pred in predictions
    ]
    return outputs


def predict_multiclass_pipe(
    texts: list[str],
    model: AutoModelForSequenceClassification,
    tokenizer: PreTrainedTokenizerBase,
    description: Optional[str],
) -> list[str]:
    if len(texts) == 0:
        return []
    pipe = pipeline(
        task="text-classification", model=model, tokenizer=tokenizer, device=0
    )
    dataset = TextDataset(texts=texts)
    predictions = list(
        tqdm(pipe(dataset, batch_size=50), total=len(dataset), desc=description)
    )
    outputs = [pred["label"] for pred in predictions]
    return outputs


def predict_multiclass_dist_pipe(
    texts: list[str],
    model: AutoModelForSequenceClassification,
    tokenizer: PreTrainedTokenizerBase,
    description: Optional[str],
) -> list[list[DistLabel]]:
    if len(texts) == 0:
        return []
    pipe = pipeline(
        task="text-classification", model=model, tokenizer=tokenizer, device=0
    )
    dataset = TextDataset(texts=texts)
    predictions = list(
        tqdm(
            pipe(dataset, batch_size=50, top_k=None),
            total=len(dataset),
            desc=description,
        )
    )
    outputs = []
    for prediction in predictions:
        example_labels = []
        for label_prediction in prediction:
            label = label_prediction["label"]
            score = label_prediction["score"]
            example_labels.append(DistLabel(label=label, score=score))
        outputs.append(example_labels)

    return outputs


def create_germeval_tsv(samples: List[TargetSample], output_file: str) -> None:
    data = []
    for sample in samples:
        data.append(sample.model_dump())

    fields = ["id", "bin_one", "bin_maj", "bin_all", "multi_maj", "disagree_bin"]

    output_path = Path("predictions/" + output_file)
    with open(output_path, "w+") as out_file:
        writer = csv.DictWriter(
            out_file, delimiter="\t", lineterminator="\n", fieldnames=fields
        )
        writer.writeheader()
        writer.writerows(data)
    logger.info(f"saved under {output_path}")


def create_germeval_tsv_st2(samples: List[ST2TargetSample], output_file: str) -> None:
    data = []
    for sample in samples:
        data.append(sample.model_dump())

    fields = [
        "id",
        "dist_bin_0",
        "dist_bin_1",
        "dist_multi_0",
        "dist_multi_1",
        "dist_multi_2",
        "dist_multi_3",
        "dist_multi_4",
    ]

    output_path = Path("predictions/" + output_file)
    with open(output_path, "w+") as out_file:
        writer = csv.DictWriter(
            out_file, delimiter="\t", lineterminator="\n", fieldnames=fields
        )
        writer.writeheader()
        writer.writerows(data)
    logger.info(f"saved under {output_path}")


def approach1(data_path: Path) -> list[TargetSample]:
    m1_model, m1_tokenizer = load_model_and_tokenizer(M1_PATH)
    m2_model, m2_tokenizer = load_model_and_tokenizer(M2_PATH)
    m3_model, m3_tokenizer = load_model_and_tokenizer(M3_PATH)
    m4_model, m4_tokenizer = load_model_and_tokenizer(M4_PATH)
    m5_model, m5_tokenizer = load_model_and_tokenizer(M5_PATH)
    m6_model, m6_tokenizer = load_model_and_tokenizer(M6_PATH)

    raw_data = read_jsonl(data_path)
    ids = [example["id"] for example in raw_data]
    texts = [example["text"] for example in raw_data]

    bin_one = predict_binary_pipe(
        texts=texts, model=m1_model, tokenizer=m1_tokenizer, description="Predicting M1"
    )

    bin_one_true: list[tuple[int, str]] = []
    bin_one_false: list[tuple[int, str]] = []

    for i, (text, bin_one_label) in enumerate(zip(texts, bin_one)):
        if bin_one_label is True:
            bin_one_true.append((i, text))
        else:
            bin_one_false.append((i, text))

    bin_one_true_texts = [x[1] for x in bin_one_true]
    m2_preds = predict_binary_pipe(
        texts=bin_one_true_texts,
        model=m2_model,
        tokenizer=m2_tokenizer,
        description="Predicting M2",
    )
    m3_preds = predict_binary_pipe(
        texts=bin_one_true_texts,
        model=m3_model,
        tokenizer=m3_tokenizer,
        description="Predicting M3",
    )

    bin_maj: list[bool] = [False] * len(bin_one)
    bin_all: list[bool] = [False] * len(bin_one)
    for i, (m2_pred, m3_pred) in enumerate(zip(m2_preds, m3_preds)):
        if m3_pred is True:
            _bin_maj = True
            _bin_all = True
        elif m2_pred is True:
            _bin_maj = True
            _bin_all = False
        else:
            _bin_maj = False
            _bin_all = False
        index = bin_one_true[i][0]
        bin_maj[index] = _bin_maj
        bin_all[index] = _bin_all

    bin_maj_true: list[tuple[int, str]] = []
    bin_maj_false: list[tuple[int, str]] = []

    for i, (text, bin_maj_label) in enumerate(zip(texts, bin_maj)):
        if bin_maj_label is True:
            bin_maj_true.append((i, text))
        else:
            bin_maj_false.append((i, text))

    bin_maj_true_texts = [x[1] for x in bin_maj_true]
    bin_maj_false_texts = [x[1] for x in bin_maj_false]

    m4_preds = predict_multiclass_pipe(
        texts=bin_maj_true_texts,
        model=m4_model,
        tokenizer=m4_tokenizer,
        description="Predicting M4",
    )
    m5_preds = predict_multiclass_pipe(
        texts=bin_maj_false_texts,
        model=m5_model,
        tokenizer=m5_tokenizer,
        description="Predicting M5",
    )

    labels: list[Union[str, None]] = [None] * len(bin_one)
    for i, m4_pred in enumerate(m4_preds):
        index = bin_maj_true[i][0]
        labels[index] = m4_pred

    for i, m5_pred in enumerate(m5_preds):
        index = bin_maj_false[i][0]
        labels[index] = m5_pred

    if any(label is None for label in labels):
        raise ValueError("not all labels are not None")

    bin_all_false: list[tuple[int, str]] = []

    for i, (text, bin_all_label) in enumerate(zip(texts, bin_all)):
        if bin_all_label is False:
            bin_all_false.append((i, text))

    bin_all_texts = [x[1] for x in bin_all_false]
    m6_preds = predict_binary_pipe(
        texts=bin_all_texts,
        model=m6_model,
        tokenizer=m6_tokenizer,
        description="Predicting M6",
    )

    bin_disagree: list[bool] = [False] * len(bin_one)

    for i, m6_pred in enumerate(m6_preds):
        index = bin_all_false[i][0]
        bin_disagree[index] = m6_pred

    target_samples = []
    for (
        id,  # noqa: A001
        bin_one_label,
        bin_maj_label,
        bin_all_label,
        label,
        disagre_bin_label,
    ) in zip(ids, bin_one, bin_maj, bin_all, labels, bin_disagree):
        target_samples.append(
            TargetSample(
                id=id,
                bin_one=bin_one_label,
                bin_maj=bin_maj_label,
                bin_all=bin_all_label,
                multi_maj=label,
                disagree_bin=disagre_bin_label,
            )
        )
    return target_samples


def approach2(data_path: Path) -> list[TargetSample]:
    m1_model, m1_tokenizer = load_model_and_tokenizer(M1_PATH)
    m4_model, m4_tokenizer = load_model_and_tokenizer(M4_PATH)
    m5_model, m5_tokenizer = load_model_and_tokenizer(M5_PATH)
    m6_model, m6_tokenizer = load_model_and_tokenizer(M6_PATH)
    m7_model, m7_tokenizer = load_model_and_tokenizer(M7_PATH)

    raw_data = read_jsonl(data_path)
    ids = [example["id"] for example in raw_data]
    texts = [example["text"] for example in raw_data]

    bin_one = predict_binary_pipe(
        texts=texts, model=m1_model, tokenizer=m1_tokenizer, description="Predicting M1"
    )

    bin_one_true: list[tuple[int, str]] = []
    bin_one_false: list[tuple[int, str]] = []

    for i, (text, bin_one_label) in enumerate(zip(texts, bin_one)):
        if bin_one_label is True:
            bin_one_true.append((i, text))
        else:
            bin_one_false.append((i, text))

    bin_one_true_texts = [x[1] for x in bin_one_true]
    m7_preds = predict_multiclass_pipe(
        texts=bin_one_true_texts,
        model=m7_model,
        tokenizer=m7_tokenizer,
        description="Predicting M7",
    )
    bin_maj = [False] * len(bin_one)
    bin_all = [False] * len(bin_one)
    for i, m7_pred in enumerate(m7_preds):
        index = bin_one_true[i][0]
        if m7_pred == "bin_maj":
            _bin_maj = True
            _bin_all = False
        elif m7_pred == "bin_all":
            _bin_maj = True
            _bin_all = True
        elif m7_pred == "no_maj":
            _bin_maj = False
            _bin_all = False
        bin_maj[index] = _bin_maj
        bin_all[index] = _bin_all

    bin_maj_true: list[tuple[int, str]] = []
    bin_maj_false: list[tuple[int, str]] = []

    for i, (text, bin_maj_label) in enumerate(zip(texts, bin_maj)):
        if bin_maj_label is True:
            bin_maj_true.append((i, text))
        else:
            bin_maj_false.append((i, text))

    bin_maj_true_texts = [x[1] for x in bin_maj_true]
    bin_maj_false_texts = [x[1] for x in bin_maj_false]

    m4_preds = predict_multiclass_pipe(
        texts=bin_maj_true_texts,
        model=m4_model,
        tokenizer=m4_tokenizer,
        description="Predicting M4",
    )
    m5_preds = predict_multiclass_pipe(
        texts=bin_maj_false_texts,
        model=m5_model,
        tokenizer=m5_tokenizer,
        description="Predicting M5",
    )

    labels: list[Union[str, None]] = [None] * len(bin_one)
    for i, m4_pred in enumerate(m4_preds):
        index = bin_maj_true[i][0]
        labels[index] = m4_pred

    for i, m5_pred in enumerate(m5_preds):
        index = bin_maj_false[i][0]
        labels[index] = m5_pred

    if any(label is None for label in labels):
        raise ValueError("not all labels are not None")

    bin_all_false: list[tuple[int, str]] = []

    for i, (text, bin_all_label) in enumerate(zip(texts, bin_all)):
        if bin_all_label is False:
            bin_all_false.append((i, text))

    bin_all_texts = [x[1] for x in bin_all_false]
    m6_preds = predict_binary_pipe(
        texts=bin_all_texts,
        model=m6_model,
        tokenizer=m6_tokenizer,
        description="Predicting M6",
    )

    bin_disagree: list[bool] = [False] * len(bin_one)

    for i, m6_pred in enumerate(m6_preds):
        index = bin_all_false[i][0]
        bin_disagree[index] = m6_pred

    target_samples = []
    for (
        id,  # noqa: A001
        bin_one_label,
        bin_maj_label,
        bin_all_label,
        label,
        disagre_bin_label,
    ) in zip(ids, bin_one, bin_maj, bin_all, labels, bin_disagree):
        target_samples.append(
            TargetSample(
                id=id,
                bin_one=bin_one_label,
                bin_maj=bin_maj_label,
                bin_all=bin_all_label,
                multi_maj=label,
                disagree_bin=disagre_bin_label,
            )
        )
    return target_samples


def parse_bin_dist(
    bin_one: list[tuple[bool, float]], num_annotators: list[int]
) -> List[Tuple[float, float]]:
    bin_dist_tuples: list[tuple[float, float]] = []
    for bin_one_tuple, _num_annotators in zip(bin_one, num_annotators):
        pred = bin_one_tuple[0]
        score: float = bin_one_tuple[1]
        possible_values: list[float] = np.linspace(
            start=0, stop=1, num=_num_annotators + 1
        ).tolist()
        min_value = min(
            possible_values,
            key=lambda possible_value: abs(possible_value - score),
        )
        value = round(
            min_value,
            ndigits=2,
        )
        neg_value: float = float(Decimal("1.0") - Decimal(f"{value}"))
        if pred is True:
            bin_dist_tuples.append((neg_value, value))
        else:
            bin_dist_tuples.append((value, neg_value))

    return bin_dist_tuples


def parse_multi_dist(
    multiclass_outs: list[list[DistLabel]], num_annotators: list[int]
) -> list[tuple[float, ...]]:
    multi_dist_tuples: list[tuple[float, ...]] = []

    for mutliclass_output, _num_annotators in zip(multiclass_outs, num_annotators):
        probabilities: list[Union[int, float]] = [0, 0, 0, 0, 0]
        for label in mutliclass_output:
            if label.label == Label.KEIN:
                index = 0
            elif label.label == Label.GERING:
                index = 1
            elif label.label == Label.VORHANDEN:
                index = 2
            elif label.label == Label.STARK:
                index = 3
            elif label.label == Label.EXTREM:
                index = 4
            probabilities[index] = label.score

        votes = [0] * len(probabilities)
        for _ in range(_num_annotators):
            vote = random.choices(range(len(probabilities)), probabilities)[0]  # noqa: S311
            votes[vote] += 1
        dist = [round(vote / _num_annotators, ndigits=2) for vote in votes]
        dist_sum = math.fsum(dist)
        if dist_sum != 1.0:
            diff = round(1.0 - dist_sum, ndigits=2)  # 1.2 = -0.2
            index = dist.index(max(dist))
            dist[index] = round(dist[index] + diff, ndigits=2)

        multi_dist_tuples.append(tuple(dist))

    return multi_dist_tuples


def subtask2(data_path: Path) -> list[ST2TargetSample]:
    raw_data = read_jsonl(data_path)
    ids = [example["id"] for example in raw_data]
    texts = [example["text"] for example in raw_data]
    annotators = [example["annotators"] for example in raw_data]
    num_annotators = [len(_annotators) for _annotators in annotators]

    m1_model, m1_tokenizer = load_model_and_tokenizer(M1_PATH)
    bin_one = predict_binary_dist_pipe(
        texts=texts, model=m1_model, tokenizer=m1_tokenizer, description="Predicting M1"
    )
    bin_dist = parse_bin_dist(bin_one=bin_one, num_annotators=num_annotators)

    m5_model, m5_tokenizer = load_model_and_tokenizer(M5_PATH)
    multiclass_outputs = predict_multiclass_dist_pipe(
        texts=texts, model=m5_model, tokenizer=m5_tokenizer, description="Predicting M5"
    )

    multi_dist_outputs = parse_multi_dist(
        multiclass_outs=multiclass_outputs, num_annotators=num_annotators
    )

    outputs = []
    for _id, (dist_bin_0, dist_bin_1), (
        dist_m_0,
        dist_m_1,
        dist_m_2,
        dist_m_3,
        dist_m_4,
    ) in zip(ids, bin_dist, multi_dist_outputs):
        outputs.append(
            ST2TargetSample(
                id=_id,
                dist_bin_0=dist_bin_0,
                dist_bin_1=dist_bin_1,
                dist_multi_0=dist_m_0,
                dist_multi_1=dist_m_1,
                dist_multi_2=dist_m_2,
                dist_multi_3=dist_m_3,
                dist_multi_4=dist_m_4,
            )
        )

    return outputs


def main() -> None:
    parser = get_argparser()
    args = parser.parse_args()
    output_file = args.output
    approach = args.approach

    logging.info(f"output_file: {output_file}")
    logging.info(f"approach: {approach}")

    if approach is None:
        task = "st2"
    elif approach >= 3 or approach <= 0:
        raise ValueError("Approach must be 1 or 2")
    else:
        task = "st1"

    if task == "st1":
        if approach == 1:
            samples = approach1(data_path=DATA_PATH)
            create_germeval_tsv(samples=samples, output_file=output_file)
        elif approach == 2:
            samples = approach2(data_path=DATA_PATH)
            create_germeval_tsv(samples=samples, output_file=output_file)
    elif task == "st2":
        st2_samples = subtask2(DATA_PATH)
        create_germeval_tsv_st2(samples=st2_samples, output_file=output_file)


if __name__ == "__main__":
    main()
