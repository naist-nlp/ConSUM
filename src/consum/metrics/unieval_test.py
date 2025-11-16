import pytest
import torch

from .unieval import MetricUniEval
from UniEval.metric.evaluator import get_evaluator
from UniEval.utils import convert_to_json

from ..utils.summ_test_set import HYPOTHESES, REFERENCES, SOURCES, UNIEVAL_SCORES

class TestMetricUniEval:
    @pytest.fixture(scope="class")
    def metric_unieval(self):
        return MetricUniEval(MetricUniEval.Config())

    @pytest.fixture(scope="class")
    def unieval_scorer(self):
        return get_evaluator(**MetricUniEval.Config().__dict__)
    
    def test_score(self, metric_unieval: MetricUniEval):
        metric_scores = []
        for hyp, ref, src in zip(HYPOTHESES, REFERENCES, SOURCES):
            metric_score = metric_unieval.score(hyp, ref, src)
            metric_scores.append(list(metric_score.values()))

        unieval_scores = [list(d.values()) for d in UNIEVAL_SCORES]
        torch.testing.assert_close(
            torch.tensor(metric_scores, dtype=torch.float32),
            torch.tensor(unieval_scores, dtype=torch.float32),
        )    

    def test_scores(self, metric_unieval: MetricUniEval):
        metric_scores = metric_unieval.scores(HYPOTHESES, REFERENCES, SOURCES)
        metric_scores = [list(d.values()) for d in metric_scores]

        unieval_scores = [list(d.values()) for d in UNIEVAL_SCORES]

        torch.testing.assert_close(
            torch.Tensor(metric_scores),
            torch.Tensor(unieval_scores),
        )

    def test_pairwise_scores(self, metric_unieval: MetricUniEval, unieval_scorer):
        pairwise_scores = metric_unieval.pairwise_scores(HYPOTHESES, REFERENCES, SOURCES)
        
        unieval_scores = []
        for hyp in HYPOTHESES:
            data = convert_to_json(
                output_list=[hyp] * len(REFERENCES),
                ref_list=REFERENCES,
                src_list=SOURCES                   
            )
            scores = unieval_scorer.evaluate(data)
            for score in scores:
                if "overall" in score.keys():
                    unieval_scores.append(score["overall"])
                else:
                    unieval_scores.append(list(score.values())[0])  # Get the first score if overall not present

        torch.testing.assert_close(
            pairwise_scores,
            torch.tensor(unieval_scores, dtype=torch.float32).to(metric_unieval.device),
        )

    def test_pairwise_scores_empty_inputs(self, metric_unieval: MetricUniEval, unieval_scorer):
        hyps = ["", "this is a test", ""]
        refs = ["", "this is a fest", ""]
        srcs = ["source one", "source two", "source three"]
        pairwise_scores = metric_unieval.pairwise_scores(hyps, refs, srcs)
        
        unieval_scores = []
        for hyp in hyps:
            data = convert_to_json(
                output_list=[hyp if hyp.strip() != "" else "No text given"] * len(refs),
                ref_list=refs,
                src_list=srcs
            )
            scores = unieval_scorer.evaluate(data)
            for score in scores:
                if "overall" in score.keys():
                    unieval_scores.append(score["overall"])
                else:
                    unieval_scores.append(list(score.values())[0])  # Get the first score if overall not present

        torch.testing.assert_close(
            pairwise_scores,
            torch.tensor(unieval_scores, dtype=torch.float32).to(metric_unieval.device),
        )
