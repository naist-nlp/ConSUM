from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from torch import Tensor
from menli.MENLI import MENLI
from mbrs.metrics.base import Metric, register

@register("menli")
class MetricMENLI(Metric):
    """
    Metric class for MENLI utility function for MBR decoding.
    """

    HIGHER_IS_BETTER: bool = True

    @dataclass
    class Config(Metric.Config):
        normalized: bool = False
        direction: str = "hr"
        formula: str = "e-c"
        nli_weight: float = 1.0
        combine_with: str = "None"
        model: str = "D"
    
    cfg: MetricMENLI.Config

    def __init__(self, cfg: MetricMENLI.Config):
        super().__init__(cfg)
        self.scorer : MENLI = MENLI(
            direction=cfg.direction,
            formula=cfg.formula,
            nli_weight=cfg.nli_weight,
            combine_with=cfg.combine_with,
            model=cfg.model,
        )

    def pairwise_scores(
        self, hypotheses: list[str], references: list[str], sources: Optional[list[str]] = None
    ) -> Tensor:
        if sources is not None:
            return Tensor(
                [self.scores([hyp] * len(references), references, [src] * len(references)) for hyp, src in zip(hypotheses, sources)]
            )
        else:
            return Tensor(
                [self.scores([hyp] * len(references), references) for hyp in hypotheses]
            )

    def scores(
        self, hypotheses: list[str], references: list[str], sources: Optional[list[str]] = None
    ) -> list[float]:
        if sources is not None:
            score = self.scorer.score_nli(refs=references, hyps=hypotheses, srcs=sources)
        else:
            score = self.scorer.score_nli(refs=references, hyps=hypotheses)
        return score
    
    def score(
        self, hypothesis: str, reference: str, source: Optional[str] = None
    ) -> float:
        if source is not None:
            score = self.scorer.score_nli(refs=[reference], hyps=[hypothesis], srcs=[source])
        else:
            score = self.scorer.score_nli(refs=[reference], hyps=[hypothesis])
        return float(score[0])