# Required 2 things from simcls: preprocessed data and trained model
# Preprocessed data can be generated via SimCLS/custom_data_preprocess.py
# Trained model can be obtained by training using the preprocessed data via SimCLS/main.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from transformers import RobertaTokenizer
from mbrs.metrics.base import MetricReferenceless, register
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from functools import partial
from huggingface_hub import hf_hub_download

from ..modules import SimCLSScorer
from ..modules.simcls.data_utils import ReRankingDataset, collate_mp, to_cuda

@register("simcls")
class MetricSimCLS(MetricReferenceless):
    """
    Metric class for SimCLS reference-free metric.
    """

    HIGHER_IS_BETTER: bool = True

    @dataclass
    class Config(MetricReferenceless.Config):
        simcls_path: Optional[str] = None
        model_type : str = "roberta-base"
        batch_size: int = 8
        device: str = "cuda"
        num_samples: int = 0

    cfg: MetricSimCLS.Config

    def __init__(self, cfg: MetricSimCLS.Config):
        super().__init__(cfg)
        self.tokenizer: RobertaTokenizer = RobertaTokenizer.from_pretrained(cfg.model_type)
        self.scorer : SimCLSScorer = SimCLSScorer(self.cfg.model_type, self.tokenizer.pad_token_id)
        if cfg.simcls_path is not None:
            weight_file_path = hf_hub_download(repo_id=cfg.simcls_path, filename="scorer.bin")
            self.scorer.load_state_dict(torch.load(weight_file_path, map_location=self.cfg.device))
            self.scorer.eval()
        else:
            raise ValueError("Please provide the huggingface path to the trained SimCLS model via 'simcls_path' in the config.")

    def _load_data(self, candidate_cnt: int, data_path: str):
        """
        Loads the data into a DataLoader for evaluation.
        Args:
            data: Path to the preprocessed data for SimCLS.
        Returns:
            A DataLoader object for the evaluation data.
        """
        
        collate_fn = partial(collate_mp, pad_token_id=self.tokenizer.pad_token_id, is_test=True)
        dataset = ReRankingDataset(data_path, self.cfg.model_type, is_test=True, maxlen=512, 
                                is_sorted=False, is_untok=True, maxnum=candidate_cnt)
        dataloader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=False, collate_fn=collate_fn)
        self.num_samples = len(dataset)
        print(f"Loaded {self.num_samples} samples for SimCLS evaluation.")
        return dataloader

    def scores(
        self, data_path: str, candidate_cnt: int
    ) -> list[float]:
        data = self._load_data(candidate_cnt=candidate_cnt, data_path=data_path)
        
        scores = []
        with torch.no_grad():
            for batch in tqdm(data, desc="Scoring batches", total=len(data)):
                to_cuda(batch, self.device)
                src_input_ids = batch['src_input_ids']
                candidate_ids = batch['candidate_ids']
                ref_input_ids = batch['tgt_input_ids']
                
                output = self.scorer(src_input_ids, candidate_ids, ref_input_ids)
                similarity = output['score'].cpu().numpy()
                similarity = similarity.flatten()
                scores.extend(similarity.tolist())
        
        return scores

    def score(
        self, hypothesis: str, source: str
    ) -> dict:
        raise NotImplementedError("Use the 'scores' method with a data path for batch scoring.")