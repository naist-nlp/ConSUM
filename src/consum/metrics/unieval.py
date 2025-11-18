from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from torch import Tensor

from mbrs.metrics.base import Metric, register

from ..modules import UniEvalScorer
from ..modules.unieval.utils import convert_to_json

@register("unieval")
class MetricUniEval(Metric):
    """
    Metric class for UniEval utility function for MBR decoding.
    """

    HIGHER_IS_BETTER: bool = True

    @dataclass
    class Config(Metric.Config):
        task: str = "summarization"
        max_length: int = 1024
        device: str = "cuda"
        cache_dir: Optional[str] = None
    
    cfg: MetricUniEval.Config

    def __init__(self, cfg: MetricUniEval.Config):
        super().__init__(cfg)
        self.scorer = UniEvalScorer(
            task=cfg.task,
            max_length=cfg.max_length,
            device=cfg.device,
            cache_dir=cfg.cache_dir
        )

    def handle_missing_hypothesis(
        self, hypotheses: list[str]
    ) -> list[str]:
        return [hyp if hyp.strip() != "" else "No text given" for hyp in hypotheses]

    def pairwise_scores(
        self, hypotheses: list[str], references: list[str], sources: list[str]
    ) -> Tensor:
        hypotheses = self.handle_missing_hypothesis(hypotheses)
        overall_scores = []
        for hyp in hypotheses:
            scores = self.scores([hyp] * len(references), references, sources)
            
            for score in scores:
                if "overall" in score.keys():
                    overall_scores.append(score["overall"])
                else:
                    overall_scores.append(list(score.values())[0])  # Get the first score if overall not present
        return Tensor(overall_scores)

    def scores(
        self, hypotheses: list[str], references: list[str], sources: list[str]
    ) -> list[dict]:
        hypotheses = self.handle_missing_hypothesis(hypotheses)
        data = convert_to_json(
            output_list=hypotheses, 
            ref_list=references, 
            src_list=sources
        )
        scores = self.scorer.evaluate(data)
        return scores
    
    def score(
        self, hypothesis: str, reference: str, source: str
    ) -> dict:
        hypothesis = self.handle_missing_hypothesis([hypothesis])[0]
        data = convert_to_json(
            output_list=[hypothesis], 
            ref_list=[reference], 
            src_list=[source]
        )
        score = self.scorer.evaluate(data)
        return score[0]