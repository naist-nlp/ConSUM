import pytest
import torch
from torch import Tensor
import numpy as np

from mbrs.metrics import Metric, MetricReferenceless
from mbrs.selectors import Selector
from mbrs.conftest import *

from .mixed_ref_mbr import DecoderMixedRefMBR
from ..metrics import MetricMENLI, MetricFENICE, MetricFIZZ, MetricROUGE

from ..utils.summ_test_set import HYPOTHESES, REFERENCES, SOURCES, ROUGE_SCORES, MENLI_SCORES, FENICE_SCORES, FIZZ_SCORES

REFERENCE_FREE_METRICS = [
    MetricFENICE(MetricFENICE.Config()),
    # MetricFIZZ(MetricFIZZ.Config()),
]
REFERENCE_BASED_METRICS = [
    MetricMENLI(MetricMENLI.Config()),
    MetricROUGE(MetricROUGE.Config()),
]

class TestDecoderMixedRefMBR:

    def _zscore_normalize(self, 
        scores: Tensor, 
        epsilon: float
    ) -> Tensor:
        mean = scores.mean()
        std = scores.std()
        return (scores - mean) / (std + epsilon)

    @pytest.mark.parametrize("metric_ref_based, metric_refless", [(ref_based, ref_free) for ref_based in REFERENCE_BASED_METRICS for ref_free in REFERENCE_FREE_METRICS])
    def test_decode(self, metric_ref_based: Metric, metric_refless: MetricReferenceless):
        decoder = DecoderMixedRefMBR(
            DecoderMixedRefMBR.Config(
                w_mbr=0.5, 
                w_referenceless=0.5, 
                epsilon=1e-8,
            ),
            metric_ref_based=metric_ref_based,
            metric_referenceless=metric_refless,
        )
        output = decoder.decode(
            HYPOTHESES,
            REFERENCES,
            SOURCES[0],
            nbest=len(HYPOTHESES)
        )

        if type(metric_ref_based) == MetricROUGE:
            ref_based_scores = [np.array(list(d.values())).mean() for d in ROUGE_SCORES]
        elif type(metric_ref_based) == MetricMENLI:
            ref_based_scores = [d["menli_summ"] for d in MENLI_SCORES]
        ref_based_scores = self._zscore_normalize(torch.tensor(ref_based_scores), epsilon=1e-8)

        if type(metric_refless) == MetricFENICE:
            referenceless_scores = [d["fenice_score"] for d in FENICE_SCORES]
        elif type(metric_refless) == MetricFIZZ:
            referenceless_scores = [d["fizz_score"] for d in FIZZ_SCORES]
        referenceless_scores = self._zscore_normalize(torch.tensor(referenceless_scores), epsilon=1e-8)

        expected_scores = [0.5 * ref + 0.5 * ref_free for ref, ref_free in zip(ref_based_scores, referenceless_scores)]
        expected_scores = sorted(expected_scores, reverse=True)

        print("Output scores:", output.score)
        print("Expected scores:", expected_scores)

        torch.testing.assert_close(
            torch.tensor(output.score, dtype=torch.float32),
            torch.tensor(expected_scores, dtype=torch.float32),
        )

