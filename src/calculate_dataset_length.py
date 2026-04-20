import json
from generate import get_model
import os
import argparse

datasets = ["cnn_dm", "xsum"]
settings = ["simcls-gen"]
models = ["bart", "pegasus", "t5-large", "llama-3"]

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate dataset lengths")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset without extension (e.g., /path/to/dataset for dataset.src and dataset.tgt)",
    )
    parser.add_argument(
        "--cand_path",
        type=str,
        required=True,
        help="Path to the candidate file (can be .txt or .json)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the output JSON file with lengths",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model",
    )
    parser.add_argument(
        "--cand_count",
        type=int,
        required=True,
        help="Number of candidates per source",
    )
    return parser.parse_args()

def calculate_lengths(output_path, cand_path, dataset_path, model_name, cand_count):
    if os.path.exists(output_path):
        print(f"Output file {output_path} already exists")
        return
    rows = []
    with open(f"{dataset_path}.src", "r") as f:
        srcs = f.readlines()
    with open(f"{dataset_path}.tgt", "r") as f:
        tgts = f.readlines()
    if cand_path.endswith(".txt"):
        with open(cand_path, "r") as f:
            lines = f.readlines()
        for i in range(len(srcs)):
            for j in range(cand_count):
                rows.append({
                    "source": srcs[i],
                    "target_summ": tgts[i],
                    "hypothesis": lines[i * cand_count + j].strip(),
                })
    elif cand_path.endswith(".json"):
        with open(cand_path, "r") as f:
            candidates = json.load(f)
        for i in range(len(candidates)):
            for j in range(cand_count):
                rows.append({
                    "source": srcs[i],
                    "target_summ": tgts[i],
                    "hypothesis": candidates[i]['candidates'][j]['text'],
                })

    tokenizer, _ = get_model(model_name, dataset_path.split("/")[-1])
    res = []
    for row in rows:
        source_len = len(tokenizer(row["source"], add_special_tokens=True)["input_ids"])
        target_len = len(tokenizer(row["target_summ"], add_special_tokens=True)["input_ids"])
        hyp_len = len(tokenizer(row["hypothesis"], add_special_tokens=True)["input_ids"])
        res.append({
            "source": row["source"],
            "target_summ": row["target_summ"],
            "hypothesis": row["hypothesis"],
            "source_len": source_len,
            "target_len": target_len,
            "hyp_len": hyp_len,
        })
    with open(output_path, "w") as f:
        json.dump(res, f, indent=4)

if __name__ == "__main__":
    args = parse_args()
    calculate_lengths(args.output_path, args.cand_path, args.dataset_path, args.model_name, args.cand_count)