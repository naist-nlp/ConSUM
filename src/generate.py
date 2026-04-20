import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import set_seed, PreTrainedTokenizer, PreTrainedModel
from datasets import Dataset
import argparse

seed_number = 42

def parser():
    """
    Parse command line arguments for sampling candidates.
    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Sample candidates using a language model for summarization tasks.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path of the file dataset to use")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--output_path", type=str, required=True, help="Output path to save the sampled candidates")
    parser.add_argument("--candidate_count", type=int, default=4, help="Number of candidates to sample per source")
    parser.add_argument("--decoding_method", type=str, default="greedy", choices=["epsilon", "ancestral", "diverse-beam"], help="Decoding method to use")
    parser.add_argument("--eps_val", type=float, default=0.1, help="Epsilon value for MAP decoding")
    parser.add_argument("--num_beams", type=int, default=5, help="Number of beams for diverse beam search")
    parser.add_argument("--num_beam_groups", type=int, default=2, help="Number of beam groups for diverse beam search")
    parser.add_argument("--format", type=str, default="json", choices=["json", "txt"], help="Output format for the candidates")
    return parser.parse_args()

def get_model(
    model_name: str, dataset_name: str
) -> tuple[PreTrainedTokenizer, PreTrainedModel]:
    """
    Load a model and tokenizer based on the model name and dataset name.
    Args:
        model_name (str): Name of the model to load.
        dataset_name (str): Name of the dataset to load the model for.
    Returns:
        tokenizer (AutoTokenizer): Tokenizer for the model.
        model (AutoModelForSeq2SeqLM or AutoModelForCausalLM): Model
    Raises:
        ValueError: If the model or dataset name is not recognized.
    """
    tokenizer, model = None, None
    if model_name == "bart":
        if dataset_name == "cnn_dm":
            tokenizer = AutoTokenizer.from_pretrained(
                "facebook/bart-large-cnn", cache_dir=os.environ["HF_HUB_CACHE"]
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                "facebook/bart-large-cnn", cache_dir=os.environ["HF_HUB_CACHE"]
            )
        elif dataset_name == "xsum":
            tokenizer = AutoTokenizer.from_pretrained(
                "facebook/bart-large-xsum", cache_dir=os.environ["HF_HUB_CACHE"]
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                "facebook/bart-large-xsum", cache_dir=os.environ["HF_HUB_CACHE"]
            )
        else:
            raise ValueError(f"Model {model_name}-{dataset_name} not found")
    elif model_name == "pegasus":
        if dataset_name == "cnn_dm":
            tokenizer = AutoTokenizer.from_pretrained(
                "google/pegasus-cnn_dailymail", cache_dir=os.environ["HF_HUB_CACHE"]
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                "google/pegasus-cnn_dailymail", cache_dir=os.environ["HF_HUB_CACHE"]
            )
        elif dataset_name == "xsum":
            tokenizer = AutoTokenizer.from_pretrained(
                "google/pegasus-xsum", cache_dir=os.environ["HF_HUB_CACHE"]
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                "google/pegasus-xsum", cache_dir=os.environ["HF_HUB_CACHE"]
            )
        else:
            raise ValueError(f"Model {model_name}-{dataset_name} not found")
    elif model_name == "t5-large":
        if dataset_name == "cnn_dm":
            tokenizer = AutoTokenizer.from_pretrained("naist-nlp/t5-large_cnn-dm")
            model = AutoModelForSeq2SeqLM.from_pretrained("naist-nlp/t5-large_cnn-dm")
        elif dataset_name == "xsum":
            tokenizer = AutoTokenizer.from_pretrained("naist-nlp/t5-large_xsum")
            model = AutoModelForSeq2SeqLM.from_pretrained("naist-nlp/t5-large_xsum")
        else:
            raise ValueError(f"Model {model_name}-{dataset_name} not found")
    elif model_name == "llama-3":
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            padding_side="left",
            cache_dir=os.environ["HF_HUB_CACHE"],
        )
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=os.environ["HF_HUB_CACHE"],
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        raise ValueError(f"Model {model_name} not found")
    return tokenizer, model


def compute_probability_s2s(
    tokenizer: PreTrainedTokenizer, candidate_output
) -> np.ndarray:
    """
    This compute_prob function is compatible with seq2seq models.
    Doesn't work on language models.
    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer used for the model.
        candidate_output (torch.nn.Module): The output from the model's generate method.
    Returns:
        np.ndarray: An array of probabilities for each candidate sequence.
    """
    batch_size = candidate_output.sequences.shape[0]
    probs = np.array([1.0] * batch_size)
    # terms = [False] * batch_size
    for i in range(len(candidate_output.scores)):
        p = np.array([1.0] * batch_size)
        for b in range(batch_size):
            if hasattr(tokenizer, "pad_token_id"):
                if (
                    candidate_output.sequences[b][i + 1] == tokenizer.pad_token_id
                    or candidate_output.sequences[b][i + 1] == tokenizer.eos_token_id
                ):
                    continue
            log_probs = torch.nn.functional.log_softmax(
                candidate_output.scores[i][b], dim=-1
            )
            p[b] = torch.exp(log_probs[candidate_output.sequences[b][i + 1]])
        probs *= p
    return probs


def sample_llm_candidates(
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    candidate_count: int,
    torch_device: torch.device,
    src_dataset: Dataset,
    dataset_name: str,
    model_name: str,
    output_path: str,
    generation_kwargs: dict,
) -> list:
    """
    Sample candidates using a language model (LLM) for summarization tasks.
    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer for the model.
        model (PreTrainedModel): The model to generate candidates.
        candidate_count (int): Number of candidates to generate per source.
        torch_device (torch.device): The device to run the model on.
        src_dataset (Dataset): The source dataset containing text to summarize.
        dataset_name (str): Name of the dataset.
        model_name (str): Name of the model.
        generation_kwargs (dict): Additional arguments for generation.
    Returns:
        None: Saves the generated candidates to a JSON file.
    """
    set_seed(seed_number)

    all_candidates = []
    for i in tqdm(
        range(src_dataset.num_rows),
        desc=f"Sampling {dataset_name} using {model_name}",
        mininterval=150,
    ):
        src = src_dataset[i]["source"]
        messages = [
            {
                "role": "system",
                "content": "You are an assistant who replies with a summary to every message.",
            },
            {"role": "user", "content": f"Summarize the following text: \n\n {src}"},
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids = input_ids.to(torch_device)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        candidates = []
        candidate_output = model.generate(
            **input_ids,
            **generation_kwargs,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=500,
        )
        if generation_kwargs["num_beams"] > 1:
            probs = candidate_output.sequences_scores.cpu().numpy().astype(np.float64)
        else:
            probs = compute_probability_s2s(tokenizer, candidate_output)
        for j in range(candidate_count):
            candidate_text = tokenizer.decode(
                candidate_output.sequences[j], skip_special_tokens=True
            )
            candidate_text = candidate_text.split("\n\n")[-1]
            candidates.append({"text": candidate_text, "prob": probs[j]})
        assert len(candidates) > 0, "No candidates generated"
        all_candidates.append({"src": src, "candidates": candidates})
    assert (
        len(all_candidates) == src_dataset.num_rows
    ), f"Length of all_candidates {len(all_candidates)} not equal to dataset length {src_dataset.num_rows}"
    assert (
        len(all_candidates[0]["candidates"]) == candidate_count
    ), f"Length of candidates {len(all_candidates[0]['candidates'])} not equal to candidate_count {candidate_count}"

    return all_candidates


def sample_candidates(
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    candidate_count: int,
    torch_device: torch.device,
    src_dataset: Dataset,
    dataset_name: str,
    model_name: str,
    output_path: str,
    generation_kwargs: dict,
) -> list:
    """
    Sample candidates using a seq2seq model for summarization tasks.
    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer for the model.
        model (PreTrainedModel): The model to generate candidates.
        candidate_count (int): Number of candidates to generate per source.
        torch_device (torch.device): The device to run the model on.
        src_dataset (Dataset): The source dataset containing text to summarize.
        dataset_name (str): Name of the dataset.
        model_name (str): Name of the model.
        generation_kwargs (dict): Additional arguments for generation.
    Returns:
        None: Saves the generated candidates to a JSON file."""
    set_seed(seed_number)

    all_candidates = []
    for i in tqdm(
        range(src_dataset.num_rows),
        desc=f"Sampling {dataset_name} using {model_name}",
        mininterval=150,
    ):
        src = src_dataset[i]["source"]
        if dataset_name == "xsum":
            src_input = tokenizer(
                src,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding="longest",
            ).to(torch_device)
        elif dataset_name == "cnn_dm":
            src_input = tokenizer(
                src,
                return_tensors="pt",
                max_length=1024,
                truncation=True,
                padding="longest",
            ).to(torch_device)

        candidates = []
        candidate_output = model.generate(
            **src_input,
            **generation_kwargs,
        )
        if generation_kwargs["num_beams"] > 1:
            probs = candidate_output.sequences_scores.cpu().numpy().astype(np.float64)
        else:
            probs = compute_probability_s2s(tokenizer, candidate_output)
        for j in range(candidate_count):
            candidate_text = tokenizer.decode(
                candidate_output.sequences[j], skip_special_tokens=True
            )
            candidates.append({"text": candidate_text, "prob": probs[j]})
        assert len(candidates) > 0, "No candidates generated"
        all_candidates.append({"src": src, "candidates": candidates})
    assert (
        len(all_candidates) == src_dataset.num_rows
    ), f"Length of all_candidates {len(all_candidates)} not equal to dataset length {src_dataset.num_rows}"
    assert (
        len(all_candidates[0]["candidates"]) == candidate_count
    ), f"Length of candidates {len(all_candidates[0]['candidates'])} not equal to candidate_count {candidate_count}"

    return all_candidates

def save_output(candidates, candidate_count, output_path, format):
    dir_path = os.path.dirname(output_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    if format == "txt":
        with open(output_path, "w") as f:
            for item in candidates:
                for cand in item['candidates']:
                    f.write(f"{cand['text']}\n")
    elif format == "json":
        with open(os.path.join(dir_path, f"candidate_{candidate_count}.json"), "w") as f:
            json.dump(candidates, f, indent=4)


if __name__ == "__main__":
    args = parser()

    dataset_path = args.dataset_path
    dataset_name = dataset_path.split("/")[-1].split("_")[0]
    dataset_name = "cnn_dm" if "cnn" in dataset_name else dataset_name
    model_name = args.model
    candidate_count = args.candidate_count
    decoding_method = args.decoding_method

    output_path = args.output_path

    generation_kwargs = {}
    generation_kwargs["num_return_sequences"] = candidate_count
    generation_kwargs["output_scores"] = True
    generation_kwargs["return_dict_in_generate"] = True
    if decoding_method == "epsilon":
        generation_kwargs["epsilon_cutoff"] = args.eps_val
        generation_kwargs["do_sample"] = True
        generation_kwargs["num_beams"] = 1
    elif decoding_method == "ancestral":
        generation_kwargs["do_sample"] = True
        generation_kwargs["num_beams"] = 1
    elif decoding_method == "diverse-beam":
        generation_kwargs["do_sample"] = False
        generation_kwargs["num_beams"] = args.num_beams
        generation_kwargs["num_beam_groups"] = args.num_beam_groups
        generation_kwargs["num_return_sequences"] = args.num_beams
        candidate_count = args.num_beams
        if generation_kwargs["num_beam_groups"] == 2:
            generation_kwargs["diversity_penalty"] = 0.5
        elif generation_kwargs["num_beam_groups"] > 2:
            generation_kwargs["diversity_penalty"] = 1.0
    else:
        raise ValueError(f"Unknown decoding method: {decoding_method}")
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(f"{dataset_path}.src", "r") as f:
        src_lines = f.readlines()
    with open(f"{dataset_path}.tgt", "r") as f:
        tgt_lines = f.readlines()
    assert len(src_lines) == len(tgt_lines), "Source and target files have different number of lines"
    src_dataset = Dataset.from_dict({"source": src_lines, "target_summ": tgt_lines})

    tokenizer, model = get_model(model_name, dataset_name)
    model.to(torch_device)

    if model_name in ["llama-3"]:
        all_candidates = sample_llm_candidates(
            tokenizer,
            model,
            candidate_count,
            torch_device,
            src_dataset,
            dataset_name,
            model_name,
            output_path,
            generation_kwargs,
        )
    else:
        all_candidates = sample_candidates(
            tokenizer,
            model,
            candidate_count,
            torch_device,
            src_dataset,
            dataset_name,
            model_name,
            output_path,
            generation_kwargs,
        )

    save_output(all_candidates, candidate_count, output_path, args.format)
