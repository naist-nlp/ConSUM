# https://github.com/plm3332/FIZZ

from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from datasets import Dataset
import numpy as np
import os
import torch
from . import prompts_example
import warnings
from itertools import combinations

warnings.filterwarnings("ignore")
import nltk
from nltk.tokenize import sent_tokenize

model_map = {
    "orca2": {"model_ckpt": "microsoft/Orca-2-7b"},
    "mistral": {"model_ckpt": "mistralai/Mistral-7B-Instruct-v0.2"},
    "zephyr": {"model_ckpt": "HuggingFaceH4/zephyr-7b-beta"},
    "tals": {"model_ckpt": "tals/albert-xlarge-vitaminc-mnli"},
    "morlau": {"model_ckpt": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"}
}


class AtomicFactDecomposer:
    def __init__(self, model_name="orca2", device="cuda"):
        assert model_name in model_map.keys(), "Wrong model name: `%s`" % (model_name)

        self.model_name = model_name
        self.model_ckpt = model_map[self.model_name]["model_ckpt"]
        self.model = None
        self.device = device

    def load_lm(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_ckpt,
            use_fast=False,
            padding_side="left",
            cache_dir=os.environ["HF_HUB_CACHE"],
            device_map="auto"
        )
        # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_ckpt,
            torch_dtype=torch.float16,
            cache_dir=os.environ["HF_HUB_CACHE"],
            device_map="auto"
        )
        # self.model.to(self.device)

    def get_prompt(self, text):
        if self.model_name == "orca2":
            return prompts_example.orca_format(text)
        elif self.model_name == "mistral":
            return prompts_example.mistral_format(text)
        elif self.model_name == "zephyr":
            return prompts_example.zephyr_format(text)

    def tokenize_batch(self, texts):
        if self.model_name == "orca2":
            model_inputs = self.tokenizer(
                texts, return_tensors="pt", padding=True
            )
            return model_inputs
        else:
            model_inputs = self.tokenizer.apply_chat_template(
                texts, return_tensors="pt", padding=True
            )
            return model_inputs

    def decode_batch(self, generated_ids):
        decoded_ids = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return decoded_ids

    def generate_and_decode_batch(self, tokenized_ids):
        generated_ids = self.generate_batch(tokenized_ids)
        decoded_ids = self.decode_batch(generated_ids)
        return decoded_ids

    def generate_batch(self, input_ids):
        with torch.no_grad():
            generated_ids = self.model.generate(
                **input_ids,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return generated_ids

    def preprocess_decoded_id(self, decoded_id) -> list[str]:
        if self.model_name == "orca2":
            text = decoded_id.split("<|im_start|> assistant")[-1]
            preprocessed_text = text.replace("<|im_start|>", "")
            preprocessed_text = preprocessed_text.replace(
                "Some possible atomic facts of the text are:", ""
            )
            preprocessed_texts = preprocessed_text.split("\n-")
            preprocessed_texts = [
                text.replace("\n", "") for text in preprocessed_texts if len(text) > 0
            ]
        elif self.model_name == "zephyr":
            text = decoded_id.split("<|assistant|>")[-1]
            preprocessed_texts = text.split("\n-")
            preprocessed_texts = [
                text.replace("\n", "") for text in preprocessed_texts if len(text) > 0
            ]
        elif self.model_name == "mistral":
            text = decoded_id.split("[/INST]")[-1]
            preprocessed_texts = text.split("\n-")
            preprocessed_texts = [
                text.replace("\n", "") for text in preprocessed_texts if len(text) > 0
            ]
        return preprocessed_texts

    def batch_decompose(self, texts: list[list[str]], batch_size=8):
        if self.model == None:
            self.load_lm()

        sentences_list, summ_idx_marker = [], []
        for sentences in texts:
            temp_arr = [self.get_prompt(sentence) for sentence in sentences]
            sentences_list.extend(temp_arr)
            summ_idx_marker.append(len(sentences_list))

        input_ids = self.tokenize_batch(sentences_list)
        decoded_ids = []
        for i in tqdm(range(0, len(input_ids["input_ids"]), batch_size), desc="Decomposing"):
            batch_decoded_ids = self.generate_and_decode_batch(
                {key:value[i:i+batch_size].to(self.device) for key,value in input_ids.items()}
            )
            decoded_ids.extend(batch_decoded_ids)
        preprocessed_ids = [
            self.preprocess_decoded_id(decoded_id) for decoded_id in decoded_ids
        ]
        assert len(preprocessed_ids) == len(
            sentences_list
        ), f"Length of preprocessed ids ({len(preprocessed_ids)}) not equal to length of sentences list ({len(sentences_list)})"

        atomic_facts_list = []
        print(summ_idx_marker)
        for i in range(len(summ_idx_marker)):
            start_idx = 0 if i == 0 else summ_idx_marker[i - 1]
            end_idx = summ_idx_marker[i]
            temp_arr = []
            for j in range(start_idx, end_idx):
                temp_arr.extend(preprocessed_ids[j])
            if len(temp_arr) == 0:
                temp_arr.extend(texts[j])
            assert (
                len(temp_arr) != 0
            ), f"No atomic facts found for the text for index {start_idx}:{end_idx}, {preprocessed_ids} /// {texts}"
            atomic_facts_list.append(temp_arr)
        assert len(atomic_facts_list) == len(
            texts
        ), f"Length of atomic facts list ({len(atomic_facts_list)}) not equal to length of texts ({len(texts)})"
        return atomic_facts_list
        # return {
        #     "atomic_facts": atomic_facts_list,
        #     "decoded_ids": decoded_ids,
        #     "preprocessed_ids": preprocessed_ids,
        #     "sentences_list": sentences_list,
        #     "input_ids": input_ids.tolist(),
        # }

class AtomicFactFilterer:
    def __init__(self, model_name="tals", device="cuda"):
        assert model_name in model_map.keys(), "Wrong model name: `%s`" % (model_name)

        self.model_name = model_name
        self.model_ckpt = model_map[self.model_name]["model_ckpt"]
        self.model = None
        self.device = device

    def load_lm(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_ckpt, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_ckpt)
        self.model.to(self.device)

    def create_summ_fact_pairs(
        self, summs: list[str], atomic_facts: list[str]
    ) -> list[dict]:
        summ_fact_pairs = []
        for summ in summs:
            for fact in atomic_facts:
                summ_fact_pairs.append({"summ": summ.strip(), "fact": fact.strip()})
        return summ_fact_pairs

    def filter_condition(self, evid_score, conts_score, neuts_score):
        return evid_score > conts_score and evid_score > neuts_score

    def batch_filtering(
        self,
        summ_sentences_list: list[list[str]],
        atomic_facts_list: list[list[str]],
        batch_size=1024,
    ):
        from datasets.utils.logging import disable_progress_bar
        disable_progress_bar()

        if self.model == None:
            self.load_lm()

        summ_fact_pairs = []
        for i in range(len(summ_sentences_list)):
            row_pairs = self.create_summ_fact_pairs(
                summ_sentences_list[i], atomic_facts_list[i]
            )
            for pair in row_pairs:
                pair["row_mark"] = i
            assert len(row_pairs) == len(summ_sentences_list[i]) * len(
                atomic_facts_list[i]
            ), f"Length of row pairs ({len(row_pairs)}) not equal to length of docs sentences ({len(summ_sentences_list[i])}) and atomic facts list ({len(atomic_facts_list[i])})"
            summ_fact_pairs.extend(row_pairs)
        dataset = Dataset.from_list(summ_fact_pairs)
        
        features = self.tokenizer(
            list(dataset["summ"]) if not isinstance(dataset["summ"], list) else dataset["summ"],
            list(dataset["fact"]) if not isinstance(dataset["fact"], list) else dataset["fact"],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        self.model.eval()
        with torch.no_grad():
            evid_scores, conts_scores, neuts_scores = [], [], []
            for i in tqdm(range(0, len(dataset), batch_size), desc="Filtering"):
                features_batch = {
                    k: v[i : i + batch_size].to(self.device)
                    for k, v in features.items()
                }
                logits = self.model(**features_batch).logits
                scores = torch.nn.functional.softmax(logits, dim=-1)
                evid_scores.extend(np.array(scores[:, 0].cpu()))
                conts_scores.extend(np.array(scores[:, 1].cpu()))
                neuts_scores.extend(np.array(scores[:, 2].cpu()))

        dataset = dataset.add_column("evid_scores", evid_scores)
        dataset = dataset.add_column("conts_scores", conts_scores)
        dataset = dataset.add_column("neuts_scores", neuts_scores)
        filtered_dataset = dataset.filter(
            lambda x: self.filter_condition(
                x["evid_scores"], x["conts_scores"], x["neuts_scores"]
            ),
        )

        filtered_atomic_facts_list = []
        for row_idx in set(dataset["row_mark"]):
            filtered_sentences = filtered_dataset.filter(
                lambda x: x["row_mark"] == row_idx
            )["fact"]
            if len(filtered_sentences) == 0:
                filtered_sentences = dataset.filter(lambda x: x["row_mark"] == row_idx)[
                    "summ"
                ]
            assert (
                len(filtered_sentences) != 0
            ), f"No atomic facts found for the text for index {row_idx}"
            no_dup_list = set()
            seen_add = no_dup_list.add
            filtered_sentences = [sent for sent in filtered_sentences if not (sent in no_dup_list or seen_add(sent))]
            filtered_atomic_facts_list.append(filtered_sentences)
        assert len(filtered_atomic_facts_list) == len(
            summ_sentences_list
        ), f"Length of filtered atomic facts list ({len(filtered_atomic_facts_list)}) not equal to length of docs sentences list ({len(summ_sentences_list)})"
        return filtered_atomic_facts_list

class AtomicFactScorer:
    def __init__(self, model_name="tals", granularity="3G", device="cuda"):
        assert granularity in ["1G", "2G", "3G", "4G"], "Wrong granularity %s" % (
            granularity
        )
        assert model_name in model_map.keys(), "Wrong model name: `%s`" % (model_name)

        self.granularity = granularity
        self.gran = int(granularity[0]) + 1
        self.model_name = model_name
        self.model_ckpt = model_map[self.model_name]["model_ckpt"]
        self.model = None
        self.device = device

    def load_lm(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_ckpt, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_ckpt)
        self.model.to(self.device)

    def create_doc_fact_pairs(
        self, docs: list[str], atomic_facts: list[str]
    ) -> list[dict]:
        doc_fact_pairs = []
        for doc in docs:
            for fact in atomic_facts:
                doc_fact_pairs.append({"doc": doc.strip(), "fact": fact.strip()})
        return doc_fact_pairs

    def filter_condition(self, evid_score, conts_score, neuts_score):
        return evid_score > conts_score and evid_score > neuts_score
    
    def is_consecutive_by_one(self, numbers):
        for i in range(1, len(numbers)):
            if abs(numbers[i] - numbers[i-1]) != 1:
                return False
        return True

    def get_combinations(self, num_list, size, target):
        combination_list = []
        for i in range(1, size):
            combination = combinations(num_list, i)
            comb_list = list(combination)
            combination_list.extend(comb_list)
        
        possible_idx_list = []
        for combination in combination_list:
            idx_list = list(combination)
            if target in idx_list and self.is_consecutive_by_one(idx_list):
                possible_idx_list.append(idx_list)
        return possible_idx_list

    def get_score_model(self, dataset, batch_size, desc="Scoring"):
        features = self.tokenizer(
            list(dataset["doc"]) if not isinstance(dataset["doc"], list) else dataset["doc"],
            list(dataset["fact"]) if not isinstance(dataset["fact"], list) else dataset["fact"],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        self.model.eval()
        with torch.no_grad():
            evid_scores, conts_scores, neuts_scores = [], [], []
            for i in tqdm(range(0, len(dataset), batch_size), desc=desc):
                features_batch = {
                    k: v[i : i + batch_size].to(self.device)
                    for k, v in features.items()
                }
                logits = self.model(**features_batch).logits
                scores = torch.nn.functional.softmax(logits, dim=-1)
                evid_scores.extend(np.array(scores[:, 0].cpu()))
                conts_scores.extend(np.array(scores[:, 1].cpu()))
                neuts_scores.extend(np.array(scores[:, 2].cpu()))
        return evid_scores, conts_scores, neuts_scores

    def batch_scoring(
        self,
        docs_sentences_list: list[list[str]],
        atomic_facts_list: list[list[str]],
        batch_size=1024,
    ):
        self.load_lm()

        docs_fact_pairs = []
        for i in range(len(docs_sentences_list)):
            row_pairs = self.create_doc_fact_pairs(
                docs_sentences_list[i], atomic_facts_list[i]
            )
            for pair in row_pairs:
                pair["row_mark"] = i
            assert len(row_pairs) == len(docs_sentences_list[i]) * len(
                atomic_facts_list[i]
            ), f"Length of row pairs ({len(row_pairs)}) not equal to length of docs sentences ({len(docs_sentences_list[i])}) and atomic facts list ({len(atomic_facts_list[i])})"
            docs_fact_pairs.extend(row_pairs)
        dataset = Dataset.from_list(docs_fact_pairs)

        evid_scores, conts_scores, neuts_scores = self.get_score_model(
            dataset, batch_size
        )
        dataset = dataset.add_column("evid_scores", evid_scores)
        dataset = dataset.add_column("conts_scores", conts_scores)
        dataset = dataset.add_column("neuts_scores", neuts_scores)
        
        max_scores: list[list[float]] = [[-1.0]*len(facts) for facts in atomic_facts_list]
        rescoring_data = []
        for i, facts in enumerate(atomic_facts_list):
            for j, fact in enumerate(facts):
                temp_data = dataset.filter(
                    lambda x: x["row_mark"] == i and x["fact"] == fact
                )
                max_evid_score = max(temp_data["evid_scores"])
                max_evid_idx = temp_data["evid_scores"].index(max_evid_score)
                if self.filter_condition(
                    temp_data["evid_scores"][max_evid_idx],
                    temp_data["conts_scores"][max_evid_idx],
                    temp_data["neuts_scores"][max_evid_idx],
                ):
                    max_scores[i][j] = max_evid_score
                else:
                    expanded_gran_idx_list = self.get_combinations(
                        list(range(len(docs_sentences_list[i]))), self.gran, max_evid_idx
                    )
                    for gran_idx_list in expanded_gran_idx_list:
                        new_doc = ""
                        for gran_idx in gran_idx_list:
                            new_doc += docs_sentences_list[i][gran_idx] + " "
                        rescoring_data.append({
                            "doc": new_doc,
                            "fact": fact,
                            "row_mark": i,
                            "max_evid": max_evid_score
                        })
        if rescoring_data:
            rescoring_dataset = Dataset.from_list(rescoring_data)
            evid_rescores, _, _ = self.get_score_model(
                rescoring_dataset, batch_size, desc="Rescoring"
            )
            rescoring_dataset = rescoring_dataset.add_column("evid_scores", evid_rescores)
            for i, facts in enumerate(atomic_facts_list):
                for j, fact in enumerate(facts):
                    if max_scores[i][j] != -1.0:
                        continue
                    temp_data = rescoring_dataset.filter(
                        lambda x: x["row_mark"] == i and x["fact"] == fact
                    )
                    old_max_evid = list(set(temp_data["max_evid"]))[0]
                    new_max_evid = max(temp_data["evid_scores"])
                    max_scores[i][j] = max(old_max_evid, new_max_evid)
        assert len([score for score in max_scores[i] if score == -1.0]) == 0, f"Missing scores for atomic facts for index {i}"
        assert len(max_scores[i]) == len(facts), f"Length of max scores ({len(max_scores[i])}) not equal to length of facts ({len(facts)})"
        min_max_scores = [min(scores) for scores in max_scores]
        assert len(min_max_scores) == len(
            docs_sentences_list
        ), f"Length of min max scores ({len(min_max_scores)}) not equal to length of docs sentences list ({len(docs_sentences_list)})"

        return min_max_scores


class FizzScorer:
    def __init__(
        self,
        batch_size=32,
        decomposer_model_name="orca2",
        scoring_model_name="tals",
        granularity="3G",
        device="cuda",
    ):
        self.batch_size = batch_size
        self.device = device
        self.decomposer = AtomicFactDecomposer(model_name=decomposer_model_name)
        self.filterer = AtomicFactFilterer(model_name=scoring_model_name)
        self.scorer = AtomicFactScorer(model_name=scoring_model_name, granularity=granularity)

    def split_sentences(self, text: str) -> list[str]:
        sentences = sent_tokenize(text)
        sentences = [sent for sent in sentences if len(sent) > 10]
        if len(sentences)==0:
            sentences = [text]
        return sentences

    def batch_score(self, rows: list[dict]):
        import json
        ## Preprocess sentences
        dataset = Dataset.from_list(rows)
        dataset = dataset.map(
            lambda x: {
                "doc_sentences": self.split_sentences(x["document"]),
                "summ_sentences": self.split_sentences(x["summary"]),
            }
        )
        ## Decompose atomic facts
        atomic_facts = self.decomposer.batch_decompose(
            dataset["summ_sentences"], self.batch_size
        )
        atomic_facts = ["".join(atomic_fact) for atomic_fact in atomic_facts]
        # with open("fizz_debug_batch.json", "w") as f:
        #     json.dump(atomic_facts, f, indent=4)
        # exit(0)
        # not_same = False
        # with open("FIZZ/fizz/decompose_result.json", "r") as f:
        #     decompose_result = json.load(f)
        # for i in range(len(atomic_facts)):
        #     temp_res = " ".join(atomic_facts[i])
        #     if temp_res.strip() != decompose_result[i].strip():
        #         not_same = True
        #         print(f"Decompose result not same for index {i}")
        # with open("batch_decompose.json", "w") as f:
        #     json.dump(["".join(fact) for fact in atomic_facts], f, indent=4)
        # assert not not_same, "Decompose result not same"

        dataset = dataset.add_column("atomic_facts", atomic_facts)

        ## Filter atomic facts
        dataset = dataset.map(
            lambda x: {
                "atomic_fact_sentences": self.split_sentences(x["atomic_facts"])
            }
        )
        filtered_atomic_facts = self.filterer.batch_filtering(
            dataset["summ_sentences"], dataset["atomic_fact_sentences"],
            batch_size=self.batch_size*16
        )
        filtered_atomic_facts = [" ".join(filtered_atomic_fact) for filtered_atomic_fact in filtered_atomic_facts]
        dataset = dataset.add_column("filtered_atomic_facts", filtered_atomic_facts)

        ## Score atomic facts
        dataset = dataset.map(
            lambda x: {
                "filtered_atomic_fact_sentences": self.split_sentences(x["filtered_atomic_facts"]),
            }
        )
        scores = self.scorer.batch_scoring(
            dataset["doc_sentences"], dataset["filtered_atomic_fact_sentences"],
            batch_size=self.batch_size*16
        )
        dataset = dataset.add_column("scores", scores)
        results = [{
            "document":  doc,
            "summary": summary,
            "filtered_atomic_facts": filtered_atomic_facts,
            "score": score
        } for doc, summary, filtered_atomic_facts, score in zip(dataset["document"], dataset["summary"], dataset["filtered_atomic_facts"], dataset["scores"])]
        return results

def main():
    import argparse
    import pandas as pd
    parser = argparse.ArgumentParser(description='fairy')

    parser.add_argument('--input_path', type=str, default='data/aggre_fact_sota.csv')
    parser.add_argument('--output_path', type=str, default='data/output.csv')
    parser.add_argument('--doc_label', type=str, default='doc')
    parser.add_argument('--summary_label', type=str, default='summary')
    parser.add_argument('--atomic_facts_column', type=str, default='atomic_facts')
    parser.add_argument('--score_column', type=str, default='FIZZ_score')
    parser.add_argument('--model_name', type=str, default='orca2')
    parser.add_argument('--granularity', type=str, default='3G')

    args = parser.parse_args()

    # original = "lisa courtney, of hertfordshire, has spent most of her life collecting pokemon memorabilia."
    # atomic_facts = "Lisa Courtney is from Hertfordshire. Lisa Courtney has spent most of her life collecting Pokémon memorabilia." 
    
    dataset_input_path = args.input_path
    df = pd.read_csv(r'{}'.format(dataset_input_path), index_col = 0)
    dataset = Dataset.from_pandas(df, preserve_index=False)

    docs = dataset[args.doc_label]
    summaries = dataset[args.summary_label]

    n = dataset.num_rows

    scorer = FizzScorer(batch_size=8)
    scores = scorer.batch_score([{"document": docs[i], "summary": summaries[i]} for i in range(n)])

    filtered_atomic_facts_list = [score["filtered_atomic_facts"] for score in scores]
    score_list = [score["score"] for score in scores]
    
    dataset = dataset.add_column(args.atomic_facts_column, filtered_atomic_facts_list)
    dataset = dataset.add_column(args.score_column, score_list)
    df_output = pd.DataFrame(dataset)
    df_output.to_csv(r'{}'.format(args.output_path))

if __name__ == "__main__":
    main()
