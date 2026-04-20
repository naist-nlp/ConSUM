import os
from typing import Union
from datasets import load_dataset, Dataset, DatasetDict
import argparse


def get_dataset(
    dataset_name: str,
    subset: str,
    sample_data: int = 0,
    start_index: int = 0,
) -> Union[Dataset, DatasetDict]:
    """
    Load a dataset from Hugging Face based on the dataset name and subset.
    Args:
        dataset_name (str): Name of the dataset to load.
        subset (str): Subset of the dataset to load.
        sample_data (int): Number of samples to take from the dataset. If 0, all data is used.
        start_index (int): Starting index for sampling data.
    Returns:
        dataset (Dataset): Loaded dataset with renamed columns.
    Raises:
        ValueError: If the dataset name is not recognized.
    """
    if dataset_name == "cnn_dm":
        dataset = load_dataset(
            "abisee/cnn_dailymail", "3.0.0", cache_dir=os.environ["HF_HUB_CACHE"]
        )
        assert isinstance(dataset, dict), "Dataset should be a dictionary"

        if subset in dataset.keys():
            dataset = dataset.rename_column("article", "source")
            dataset = dataset.rename_column("highlights", "target_summ")
            dataset = (
                dataset[subset].select(range(start_index, start_index + sample_data))
                if sample_data > 0
                else dataset[subset]
            )
    elif dataset_name == "xsum":
        dataset = load_dataset(
            "EdinburghNLP/xsum", cache_dir=os.environ["HF_HUB_CACHE"]
        )
        assert isinstance(dataset, dict), "Dataset should be a dictionary"

        if subset in dataset.keys():
            dataset = dataset.rename_column("document", "source")
            dataset = dataset.rename_column("summary", "target_summ")
            dataset = (
                dataset[subset].select(range(start_index, start_index + sample_data))
                if sample_data > 0
                else dataset[subset]
            )
    else:
        raise ValueError(f"Dataset {dataset_name} not found")
    if subset == None:
        assert isinstance(
            dataset, DatasetDict
        ), f"Dataset should be of type DatasetDict, instead got {type(dataset)}"
    else:
        assert isinstance(
            dataset, Dataset
        ), f"Dataset should be of type Dataset, instead got {type(dataset)}"
    return dataset

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Get dataset from Hugging Face")
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset to load (e.g., 'cnn_dm', 'xsum')",
    )
    parser.add_argument(
        "--subset",
        type=str,
        required=True,
        help="Subset of the dataset to load (e.g., 'train', 'validation', 'test')",
    )
    parser.add_argumen(
        "--output_dir",
        type=str,
        required=True,
        help="Storing folder for the dataset"
    )
    parser.add_argument(
        "--sample_data",
        type=int,
        default=0,
        help="Number of samples to take from the dataset. If 0, all data is used.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Starting index for sampling data.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    dataset = get_dataset(
        dataset_name=args.dataset_name,
        subset=args.subset,
        sample_data=args.sample_data,
        start_index=args.start_index,
    )
    if args.sample_data == 0:
        src_filename=f"{args.dataset_name}.src"
        tgt_filename=f"{args.dataset_name}.tgt"
    elif args.sample_data > 0:
        src_filename=f"{args.dataset_name}_{args.start_index}_{args.sample_data}.src"
        tgt_filename=f"{args.dataset_name}_{args.start_index}_{args.sample_data}.tgt"
    if args.subset in args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(args.output_dir, args.subset)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, src_filename), "w") as f:
        for item in dataset["source"]:
            f.write(item.strip() + "\n")
    with open(os.path.join(output_dir, tgt_filename), "w") as f:
        for item in dataset["target_summ"]:
            f.write(item.strip() + "\n")

if __name__ == "__main__":
    main()
    