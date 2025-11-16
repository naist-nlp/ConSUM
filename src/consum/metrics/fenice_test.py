import pytest
import torch

from .fenice import MetricFENICE

from ..utils.summ_test_set import HYPOTHESES, SOURCES, FENICE_SCORES

class TestMetricFENICE:
    @pytest.fixture(scope="class")
    def metric_fenice(self):
        return MetricFENICE(MetricFENICE.Config(batch_size=4))
    
    def test_score(self, metric_fenice: MetricFENICE):
        metric_scores= []
        for hyp, src in zip(HYPOTHESES, SOURCES):
            score = metric_fenice.score(hyp, src)
            metric_scores.append(score["score"])

        fenice_scores = [d["fenice_score"] for d in FENICE_SCORES]
        torch.testing.assert_close(
            torch.Tensor(metric_scores),
            torch.Tensor(fenice_scores),
        )

    def test_scores(self, metric_fenice: MetricFENICE):
        
        scores = metric_fenice.scores(HYPOTHESES, SOURCES)
        metric_scores = scores["score"]
        
        fenice_scores = [d["fenice_score"] for d in FENICE_SCORES]

        torch.testing.assert_close(
            torch.Tensor(metric_scores),
            torch.Tensor(fenice_scores),
        )

    def test_corpus_score(self, metric_fenice: MetricFENICE):
        metric_score: float = metric_fenice.corpus_score(HYPOTHESES, SOURCES)
        
        fenice_mean_score = sum([d["fenice_score"] for d in FENICE_SCORES]) / len(FENICE_SCORES)

        torch.testing.assert_close(
            torch.Tensor([metric_score]),
            torch.Tensor([fenice_mean_score]),
        )
