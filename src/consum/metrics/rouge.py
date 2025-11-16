from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from torch import Tensor
from rouge_score import rouge_scorer
import numpy as np
from mbrs.metrics.base import Metric, register

@register("rouge")
class MetricROUGE(Metric):
    """
    Metric class for ROUGE utility function for MBR decoding.
    """

    HIGHER_IS_BETTER: bool = True

    @dataclass
    class Config(Metric.Config):
        rouge_types: list[str] = field(default_factory=lambda: ["rouge1", "rouge2", "rougeL", "rougeLsum"])
        use_stemmer: bool = False
        split_summaries: bool = False
        tokenizer: Optional[str] = None
    
    cfg: MetricROUGE.Config

    def __init__(self, cfg: MetricROUGE.Config):
        super().__init__(cfg)
        self.scorer = rouge_scorer.RougeScorer(
            rouge_types=cfg.rouge_types,
            use_stemmer=cfg.use_stemmer,
            split_summaries=cfg.split_summaries,
            tokenizer=cfg.tokenizer
        )

    def pairwise_scores(
        self, hypotheses: list[str], references: list[str]
    ) -> Tensor:
        scores = []
        for hyp in hypotheses:
            score = self.scores(
                [hyp] * len(references), references
            )
            mean_score = np.mean(
                np.array([val for val in score.values()]), axis=0
            )
            scores.append(mean_score)
        return Tensor(scores)
    
    def scores(
        self, hypotheses: list[str], references: list[str]
    ) -> dict:
        scores = {}
        for hyp, ref in zip(hypotheses, references):
            score = self.score(hyp, ref)
            for k, v in score.items():
                if k not in scores:
                    scores[k] = []
                scores[k].append(v) 
        return scores

    def score(
        self, hypothesis: str, reference: str
    ) -> dict:
        score = self.scorer.score(
            reference, hypothesis
        )
        score = {k: v.fmeasure for k, v in score.items()}
        return score
