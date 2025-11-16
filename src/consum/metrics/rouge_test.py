import pytest
import torch
import numpy as np

from .rouge import MetricROUGE
from rouge_score.rouge_scorer import RougeScorer

from ..utils.summ_test_set import HYPOTHESES, REFERENCES, ROUGE_SCORES


class TestMetricROUGE:
    @pytest.fixture(scope="class")
    def metric_rouge(self):
        return MetricROUGE(MetricROUGE.Config())

    @pytest.fixture(scope="class")
    def rouge_scorer(self) -> RougeScorer:
        return RougeScorer(**MetricROUGE.Config().__dict__)

    def test_score(self, metric_rouge: MetricROUGE):
        metric_scores = []
        for hyp, ref in zip(HYPOTHESES, REFERENCES):
            metric_score = metric_rouge.score(hyp, ref)
            metric_scores.append(list(metric_score.values()))
        
        rouge_scores = [list(d.values()) for d in ROUGE_SCORES]

        torch.testing.assert_close(
            torch.tensor(metric_scores),
            torch.tensor(rouge_scores),
        )

    def test_scores(self, metric_rouge: MetricROUGE):
        scores = metric_rouge.scores(HYPOTHESES, REFERENCES)
        metric_scores = [scores[k] for k in scores.keys()]

        rouge_scores = [list(d.values()) for d in ROUGE_SCORES]

        torch.testing.assert_close(
            torch.Tensor(metric_scores),
            torch.Tensor(rouge_scores).T,
        )

    def test_pairwise_scores(self, metric_rouge: MetricROUGE, rouge_scorer: RougeScorer):
        pairwise_scores = metric_rouge.pairwise_scores(HYPOTHESES, REFERENCES)

        rouge_scores = []
        for hyp in HYPOTHESES:
            score = []
            for ref in REFERENCES:
                sc = rouge_scorer.score(
                    ref, hyp
                )                
                score.append(np.array(list([r_score.fmeasure for r_score in sc.values()])).mean())
            rouge_scores.append(np.array(score))
        rouge_scores = torch.Tensor(rouge_scores)

        print(pairwise_scores)
        print(rouge_scores)

        torch.testing.assert_close(
            pairwise_scores,
            rouge_scores,
        )

    def test_pairwise_scores_empty_inputs(self, metric_rouge: MetricROUGE, rouge_scorer: RougeScorer):
        hyps = ["", "this is a test", ""]
        refs = ["", "this is a fest", ""]
        pairwise_scores = metric_rouge.pairwise_scores(hyps, refs)

        rouge_scores = []
        for hyp in hyps:
            score = []
            for ref in refs:
                sc = rouge_scorer.score(
                    ref, hyp
                )
                score.append(np.array(list(sc.values())).mean())
            rouge_scores.append(np.array(score))
        rouge_scores = torch.Tensor(rouge_scores)
        torch.testing.assert_close(
            pairwise_scores,
            rouge_scores,
        )
