import pytest
import torch

from .fizz import MetricFIZZ

from ..utils.summ_test_set import HYPOTHESES, SOURCES, FIZZ_SCORES

class TestMetricFIZZ:
    @pytest.fixture(scope="class")
    def metric_fizz(self):
        return MetricFIZZ(MetricFIZZ.Config(batch_size=4))
    
    def test_score(self, metric_fizz: MetricFIZZ):
        metric_scores, fizz_scores = [], []
        for hyp, src in zip(HYPOTHESES, SOURCES):
            score = metric_fizz.score(hyp, src)
            metric_scores.append(score["score"])
            
        fizz_scores = [d["fizz_score"] for d in FIZZ_SCORES]
        torch.testing.assert_close(
            torch.Tensor(metric_scores),
            torch.Tensor(fizz_scores),
        )

    def test_scores(self, metric_fizz: MetricFIZZ):
        
        scores = metric_fizz.scores(HYPOTHESES, SOURCES)
        metric_scores = scores["score"]

        fizz_scores = [d["fizz_score"] for d in FIZZ_SCORES]
        torch.testing.assert_close(
            torch.Tensor(metric_scores),
            torch.Tensor(fizz_scores),
        )

    def test_corpus_score(self, metric_fizz: MetricFIZZ):
        metric_score: float = metric_fizz.corpus_score(HYPOTHESES, SOURCES)
        fizz_scores = [d["fizz_score"] for d in FIZZ_SCORES]
        fizz_mean_score = sum(fizz_scores) / len(fizz_scores)
        torch.testing.assert_close(
            torch.tensor(metric_score),
            torch.tensor(fizz_mean_score),
        )
