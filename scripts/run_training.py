from argparse import ArgumentParser

import yaml
from germeval.classification.binary import train_binary_classifier
from germeval.classification.multiclass import train_multiclass_classifier
from germeval.dataset import (
    create_germeval24_dataset,
    filter_at_least_maj,
    filter_bin_all_is_false,
    filter_bin_one_is_true,
)
from germeval.task import Task


def get_argparser() -> ArgumentParser:
    parser = ArgumentParser("training runner")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="The path to the yaml config-file",
        required=True,
    )

    parser.add_argument(
        "-t",
        "--task",
        help="The type of task to train",
        type=Task,
        choices=list(Task),
        required=True,
    )

    parser.add_argument(
        "-i",
        "--input_data",
        help="Path (file) to training data",
        required=True,
    )

    return parser


def main() -> None:
    parser = get_argparser()
    args = parser.parse_args()
    config_file = args.config
    task = args.task
    input_path = args.input_data

    params = yaml.safe_load(open(config_file))
    train_params = params.get("train", {})
    data_params = params.get("data", {})

    dataset = create_germeval24_dataset(train_path=input_path)
    if data_params.get("filter_bin_one_is_true", False) is True:
        dataset = dataset.filter(filter_bin_one_is_true)
    if data_params.get("filter_at_least_maj", False) is True:
        dataset = dataset.filter(filter_at_least_maj)
    if data_params.get("filter_bin_all_is_false", False) is True:
        dataset = dataset.filter(filter_bin_all_is_false)

    if task == Task.GERMEVAL24:
        train_binary_classifier(dataset, train_params)

    elif task == Task.GERMEVAL24_MUTLICLASS:
        train_multiclass_classifier(dataset, train_params)


if __name__ == "__main__":
    main()
