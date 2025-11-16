from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from torch import Tensor

from mbrs import functional, timer
from mbrs.metrics import Metric, MetricReferenceless, Metrics, get_metric
from mbrs.selectors import Selector, SELECTOR_NBEST
from mbrs.decoders import register, DecoderMBR

@register("mixed_ref_mbr")
class DecoderMixedRefMBR(DecoderMBR):
    """
    MBR where there is two metrics: reference-based and referenceless.
    """

    cfg: Config

    def __init__(
        self, cfg: DecoderMixedRefMBR.Config, 
        metric_ref_based: Metric,
        metric_referenceless: Optional[Metric | MetricReferenceless] = None,
        selector: Selector = SELECTOR_NBEST,
    ) -> None:        
        super().__init__(cfg, metric_ref_based, selector=selector)

        self.metric_ref_based = self.metric
        
        if metric_referenceless:
            self.metric_referenceless = metric_referenceless
        elif cfg.referenceless_metric:
            self.metric_referenceless = get_metric(cfg.referenceless_metric)(cfg.referenceless_metric_config)
        else:
            raise ValueError("Referenceless metric must be provided.")
        
    @dataclass
    class Config(DecoderMBR.Config):
        """Configuration for the decoder.
        In addition to the base MBR config, it contains:
        - w_mbr (float): Weight for reference-based MBR.
        - w_referenceless (float): Weight for referenceless MBR.
        - epsilon (float): Small value to avoid division by zero in z-score normalization.
        """

        referenceless_metric: Metrics
        referenceless_metric_config: MetricReferenceless.Config | None = None
        w_mbr: float = 0.5  # Weight for reference-based MBR
        w_referenceless: float = 1 - w_mbr  # Weight for referenceless MBR
        epsilon: float = 1e-8  # Small value to avoid division by zero

    @dataclass
    class Output(DecoderMBR.Output):
        """
        Output class for DecoderMixedRefMBR.
        Inherits from DecoderMBR.Output.
        In addition, it can contain:
        - ref_based_scores (Optional[Tensor]): Reference-based scores that average over references.
        - referenceless_scores (Optional[Tensor]): Referenceless scores for each hypothesis to the source.
        """

        ref_based_scores: Optional[Tensor] = None
        referenceless_scores: Optional[Tensor] = None
    
    def _z_score_normalize(self, x: Tensor) -> Tensor:
        """Z-score normalization.

        Args:
            x (Tensor): Input tensor.
        Returns:
            Tensor: Z-score normalized tensor.
        """
        eps = self.cfg.epsilon
        mean = x.mean()
        std = x.std() + eps
        return (x - mean) / (std + eps)

    def _combine_scores(
        self,
        ref_scores: Tensor,
        referenceless_scores: Tensor
    ) -> Tensor:
        """Combine reference-based and referenceless scores.

        Args:
            ref_scores (Tensor): Reference-based scores.
            referenceless_scores (Tensor): Referenceless scores.
        Returns:
            Tensor: Combined scores.
        """
        assert ref_scores.shape == referenceless_scores.shape, "Shapes of reference-based and referenceless scores must match."
        
        # Normalize scores using z score normalization
        ref_scores = self._z_score_normalize(ref_scores)
        referenceless_scores = self._z_score_normalize(referenceless_scores)

        combined_scores = (
            self.cfg.w_mbr * ref_scores + 
            self.cfg.w_referenceless * referenceless_scores
        )
        return combined_scores

    def decode(
        self,
        hypotheses: list[str],
        references: list[str],
        source: str,
        nbest: int = 1,
        reference_lprobs: Optional[Tensor] = None,
    ) -> DecoderMixedRefMBR.Output:
        """Select the n-best hypotheses based on the strategy.

        Args:
            hypotheses (list[str]): Hypotheses.
            references (list[str]): References.
            source (str, optional): A source.
            nbest (int): Return the n-best hypotheses.
            reference_lprobs (Tensor, optional): Log-probabilities for each reference sample.
              The shape must be `(len(references),)`. See `https://arxiv.org/abs/2311.05263`.

        Returns:
            DecoderMixedRefMBR.Output: The n-best hypotheses.
        """
        with timer.measure("Compute reference-based scores"):
            ref_scores = self.metric_ref_based.pairwise_scores(
                hypotheses=hypotheses, 
                references=references, 
            )
            ref_scores = functional.expectation(
                ref_scores, 
                lprobs=reference_lprobs
            )

        assert ref_scores.shape[0] == len(hypotheses), "Reference-based scores length must match hypotheses length."
        
        with timer.measure("Compute referenceless scores"):
            if isinstance(self.metric_referenceless, MetricReferenceless):
                referenceless_scores = self.metric_referenceless.scores(
                    hypotheses=hypotheses, 
                    sources=[source]*len(hypotheses)
                )
            elif isinstance(self.metric_referenceless, Metric):
                referenceless_scores = self.metric_referenceless.scores(
                    hypotheses=hypotheses, 
                    references=references, 
                    sources=[source]*len(hypotheses)
                )
            else:
                raise ValueError("metric_referenceless must be either Metric or MetricReferenceless.")
                
            referenceless_scores = Tensor(referenceless_scores["score"])
        
        assert referenceless_scores.shape[0] == len(hypotheses), "Referenceless scores length must match hypotheses length."
            
        with timer.measure("Combine scores"):
            combined_scores = self._combine_scores(
                ref_scores=ref_scores,
                referenceless_scores=referenceless_scores
            )

        selector_outputs = self.select(
            hypotheses, combined_scores, nbest=nbest, source=source
        )
        return (
            self.Output(
                idx=selector_outputs.idx,
                sentence=selector_outputs.sentence,
                score=selector_outputs.score,
                ref_based_scores=ref_scores,
                referenceless_scores=referenceless_scores,
            )
        )

