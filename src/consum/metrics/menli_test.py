import pytest
import torch

from .menli import MetricMENLI
from menli.MENLI import MENLI

from ..utils.summ_test_set import HYPOTHESES, REFERENCES, MENLI_SCORES

class TestMetricMENLI:
    @pytest.fixture(scope="class")
    def metric_menli(self):
        return MetricMENLI(MetricMENLI.Config())
    
    @pytest.fixture(scope="class")
    def menli_scorer(self):
        return MENLI(
            direction="hr",
            formula="e-c",
            nli_weight=1.0,
            combine_with="None",
            model="D",
        )

    def test_score(self, metric_menli: MetricMENLI):
        metric_scores= []
        for hyp, ref in zip(HYPOTHESES, REFERENCES):
            score = metric_menli.score(hyp, ref)
            metric_scores.append(score)

        menli_scores= [d["menli_summ"] for d in MENLI_SCORES]

        torch.testing.assert_close(
            torch.tensor(metric_scores),
            torch.tensor(menli_scores),
        )

    def test_scores(self, metric_menli: MetricMENLI):
        torch.testing.assert_close(
            torch.Tensor(
                metric_menli.scores(HYPOTHESES, REFERENCES)
            ),
            torch.Tensor(
                [d["menli_summ"] for d in MENLI_SCORES]
            ),
        )

    def test_pairwise_scores(self, metric_menli: MetricMENLI, menli_scorer: MENLI):
        pairwise_scores = metric_menli.pairwise_scores(HYPOTHESES, HYPOTHESES)
        menli_scores = torch.Tensor([
            menli_scorer.score_nli(
                refs=HYPOTHESES, hyps=[hyp] * len(HYPOTHESES)
            ) for hyp in HYPOTHESES
        ])
        torch.testing.assert_close(
            pairwise_scores,
            menli_scores,
        )

    def test_pairwise_scores_empty_inputs(self, metric_menli: MetricMENLI, menli_scorer: MENLI):
        hyps = ["", "this is a test", ""]
        refs = ["", "this is a fest", ""]
        pairwise_scores = metric_menli.pairwise_scores(hyps, refs)
        menli_scores = torch.Tensor([
            menli_scorer.score_nli(
                refs=refs, hyps=[hyp] * len(refs)
            ) for hyp in hyps
        ])
        torch.testing.assert_close(
            pairwise_scores,
            menli_scores,
        )
