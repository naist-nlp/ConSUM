from __future__ import annotations
from typing import Optional

import torch
from dataclasses import dataclass, field
from torch import Tensor
from mbrs.metrics.base import Metric, register

from ..modules import MoverScorer

@register("moverscore")
class MetricMoverScore(Metric):
    """
    Metric class for MoverScore utility function for MBR decoding.
    """

    HIGHER_IS_BETTER: bool = True

    @dataclass
    class Config(Metric.Config):
        batch_size: int = 8
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
        n_gram: int = 1
        stop_words: list[str] = field(default_factory=list)
        remove_subwords: bool = True
    
    cfg: MetricMoverScore.Config

    def __init__(self, cfg: MetricMoverScore.Config):
        super().__init__(cfg)
        self.scorer = MoverScorer(
            stop_words=cfg.stop_words,
            n_gram=cfg.n_gram,
            remove_subwords=cfg.remove_subwords,
            batch_size=cfg.batch_size,
            device=cfg.device
        )

    def pairwise_scores(
        self, hypotheses: list[str], references: list[str]
    ) -> Tensor:
        return Tensor(
            [self.scores([hyp] * len(references), references) for hyp in hypotheses]
        )

    def scores(
        self, hypotheses: list[str], references: list[str], *_, **__
    ) -> list[float]:
        score = self.scorer.score(
            references, 
            hypotheses
        )
        return score
    
    def score(
        self, hypothesis: str, reference: str, *_, **__
    ) -> float:
        score = self.scorer.score(
            [reference], 
            [hypothesis]
        )
        return float(score[0])