from __future__ import annotations

from dataclasses import dataclass
from fizz.batch_fizz_score import FizzScorer
from mbrs.metrics.base import MetricReferenceless, register
from tqdm import tqdm
import torch
import numpy as np

@register("fizz")
class MetricFIZZ(MetricReferenceless):
    """
    Metric class for FIZZ reference-free metric.
    """

    HIGHER_IS_BETTER: bool = True

    @dataclass
    class Config(MetricReferenceless.Config):
        batch_size: int = 8
        backup_batch_size: int = max(1, batch_size//4)
        decomposer_model_name: str = "orca2"
        scoring_model_name: str = "tals"
        granularity: str = "3G"
        device: str = "cuda"

    cfg: MetricFIZZ.Config

    def __init__(self, cfg: MetricFIZZ.Config):
        super().__init__(cfg)
        self.scorer : FizzScorer = FizzScorer(
            batch_size=cfg.batch_size,
            decomposer_model_name=cfg.decomposer_model_name,
            scoring_model_name=cfg.scoring_model_name,
            granularity=cfg.granularity,
            device=cfg.device
        )
        self.backup_scorer : FizzScorer = FizzScorer(
            batch_size=cfg.backup_batch_size,
            decomposer_model_name=cfg.decomposer_model_name,
            scoring_model_name=cfg.scoring_model_name,
            granularity=cfg.granularity,
            device=cfg.device
        )

    def scores(
        self, hypotheses: list[str], sources: list[str]
    ) -> dict:
        temp_scores: list[dict] = []
        data = [{"document": src, "summary": hyp} for src, hyp in zip(sources, hypotheses)]
        
        for i in tqdm(
            range(0, len(data), self.cfg.batch_size), 
            desc="Calculating Fizz scores", 
            total=len(data)//self.cfg.batch_size
        ):
            batch_data = data[i:i+self.cfg.batch_size]
            try:
                res = self.scorer.batch_score(batch_data)
            except RuntimeError as e:
                print(f"Runtime error during FIZZ scoring: {e}")
                if 'out of memory' in str(e):
                    print("Out of memory error, using backup scorer with smaller batch size")
                    torch.cuda.empty_cache()
                    for j in tqdm(
                        range(0, len(batch_data), self.cfg.backup_batch_size), 
                        desc="Calculating FIZZ scores with backup scorer", 
                        total=len(batch_data)//self.cfg.backup_batch_size
                    ):
                        sub_batch_data = batch_data[j:j+self.cfg.backup_batch_size]
                        sub_res = self.backup_scorer.batch_score(sub_batch_data)
                        temp_scores.extend(sub_res)
                else:
                    raise e
            finally:
                temp_scores.extend(res)
        assert len(temp_scores) == len(hypotheses), \
            f"Length of Fizz scores {len(temp_scores)} does not match number of hypotheses {len(hypotheses)}"
        
        scores = {
            "score": [v["score"] for v in temp_scores],
            "filtered_atomic_facts": [v["filtered_atomic_facts"] for v in temp_scores]
        }
        return scores

    def score(
        self, hypothesis: str, source: str
    ) -> dict:
        score = self.scorer.batch_score(
            [{"document": source, "summary": hypothesis}]
        )[0]
        return score

    def corpus_score(self, hypotheses: list[str], sources: list[str]) -> float:
        scores = self.scores(hypotheses, sources)
        fizz_scores = scores["score"]
        return float(np.array(fizz_scores).mean())