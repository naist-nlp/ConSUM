from __future__ import annotations

import torch
from dataclasses import dataclass, field
from torch import Tensor
from moverscore_v2 import word_mover_score, get_idf_dict
from mbrs.metrics.base import Metric, register

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
        self.scorer = word_mover_score

    def pairwise_scores(
        self, hypotheses: list[str], references: list[str]
    ) -> Tensor:
        return Tensor(
            [self.scores([hyp] * len(references), references) for hyp in hypotheses]
        )

    def scores(
        self, hypotheses: list[str], references: list[str]
    ) -> list[float]:
        idf_dict_ref = get_idf_dict(references)
        idf_dict_hyp = get_idf_dict(hypotheses)
        score = self.scorer(
            refs=references,
            hyps=hypotheses,
            idf_dict_ref=idf_dict_ref,
            idf_dict_hyp=idf_dict_hyp,
            stop_words=self.cfg.stop_words,
            n_gram=self.cfg.n_gram,
            remove_subwords=self.cfg.remove_subwords,
            batch_size=self.cfg.batch_size,
            device=self.cfg.device
        )
        return score
    
    def score(
        self, hypothesis: str, reference: str
    ) -> float:
        idf_dict_ref = get_idf_dict([reference])
        idf_dict_hyp = get_idf_dict([hypothesis])
        score = self.scorer(
            refs=[reference],
            hyps=[hypothesis],
            idf_dict_ref=idf_dict_ref,
            idf_dict_hyp=idf_dict_hyp,
            stop_words=self.cfg.stop_words,
            n_gram=self.cfg.n_gram,
            remove_subwords=self.cfg.remove_subwords,
            batch_size=1,
            device=self.cfg.device
        )
        return float(score[0])