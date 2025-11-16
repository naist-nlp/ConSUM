# Since MoverScore relies on idf weights computed from the input texts,
# we compare the MetricMoverScore class against direct calls to the
# word_mover_score function from the moverscore_v2 package.

import pytest
import torch

from .moverscore import MetricMoverScore
from moverscore_v2 import word_mover_score, get_idf_dict

from ..utils.summ_test_set import HYPOTHESES, REFERENCES

class TestMetricMoverScore:
    @pytest.fixture(scope="class")
    def metric_moverscore(self):
        return MetricMoverScore(MetricMoverScore.Config(batch_size=2))

    @pytest.fixture(scope="class")
    def moverscore_score(self):
        return word_mover_score

    def test_score(self, metric_moverscore: MetricMoverScore, moverscore_score):
        metric_scores, move_scores = [], []
        for hyp, ref in zip(HYPOTHESES, REFERENCES):
            metric_score = metric_moverscore.score(hyp, ref)
            metric_scores.append(metric_score)
            move_score = moverscore_score(
                refs=[ref], 
                hyps=[hyp],
                idf_dict_ref=get_idf_dict([ref]),
                idf_dict_hyp=get_idf_dict([hyp]),
                stop_words=[],
                n_gram=1,
                remove_subwords=True,
                batch_size=1,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )[0]
            move_scores.append(float(move_score))
        
        torch.testing.assert_close(
            torch.tensor(metric_scores),
            torch.tensor(move_scores),
        )

    def test_scores(self, metric_moverscore: MetricMoverScore, moverscore_score):
        metric_scores = metric_moverscore.scores(HYPOTHESES, REFERENCES)

        mover_scores = moverscore_score(
            refs=REFERENCES,
            hyps=HYPOTHESES,
            idf_dict_ref=get_idf_dict(REFERENCES),
            idf_dict_hyp=get_idf_dict(HYPOTHESES),
            stop_words=[],
            n_gram=1,
            remove_subwords=True,
            batch_size=2,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        torch.testing.assert_close(
            torch.Tensor(
                metric_scores
            ),
            torch.Tensor(
                mover_scores
            ),
        )
    
    def test_scores_empty_inputs(self, metric_moverscore: MetricMoverScore, moverscore_score):
        hyps = ["galaxy", "", "is", "test", "apple"]
        refs = ["", "planet", "star", ".", "orange"]
        metric_scores = metric_moverscore.scores(hyps, refs)
        mover_scores = moverscore_score(
            refs=refs,
            hyps=hyps,
            idf_dict_ref=get_idf_dict(refs),
            idf_dict_hyp=get_idf_dict(hyps),
            stop_words=[],
            n_gram=1,
            remove_subwords=True,
            batch_size=2,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        avoid_idf = torch.tensor([1.0]*len(hyps), dtype=torch.double)
        
        torch.testing.assert_close(
            torch.tensor(metric_scores),
            torch.tensor(mover_scores),
        )

        assert not torch.allclose(
            torch.tensor(mover_scores),
            avoid_idf,
        ), "All scores are identical, indicating idf weights were not used."

    def test_pairwise_scores(self, metric_moverscore: MetricMoverScore, moverscore_score):
        pairwise_scores = metric_moverscore.pairwise_scores(HYPOTHESES, REFERENCES)
        
        moverscore_scores = []
        for hyp in HYPOTHESES:
            moverscore_scores.append(
                torch.Tensor(
                    moverscore_score(
                        refs=REFERENCES, hyps=[hyp] * len(REFERENCES),
                        idf_dict_ref=get_idf_dict(REFERENCES),
                        idf_dict_hyp=get_idf_dict([hyp] * len(REFERENCES)),
                        stop_words=[],
                        n_gram=1,
                        remove_subwords=True,
                        batch_size=2,
                        device="cuda" if torch.cuda.is_available() else "cpu",
                    )
                )
            )

        moverscore_scores = torch.stack(moverscore_scores)

        torch.testing.assert_close(
            pairwise_scores,
            moverscore_scores.to(metric_moverscore.device),
        )

    def test_pairwise_scores_empty_inputs(self, metric_moverscore: MetricMoverScore, moverscore_score):
        hyps = ["galaxy", "", "is", "test", "apple"]
        refs = ["", "planet", "star", ".", "orange"]
        pairwise_scores = metric_moverscore.pairwise_scores(hyps, refs)
        
        moverscore_scores = []
        for hyp in hyps:
            moverscore_scores.append(
                torch.Tensor(
                    moverscore_score(
                        refs=refs, hyps=[hyp] * len(refs),
                        idf_dict_ref=get_idf_dict(refs),
                        idf_dict_hyp=get_idf_dict([hyp] * len(refs)),
                        stop_words=[],
                        n_gram=1,
                        remove_subwords=True,
                        batch_size=2,
                        device="cuda" if torch.cuda.is_available() else "cpu",
                    )
                )
            )

        moverscore_scores = torch.stack(moverscore_scores)

        torch.testing.assert_close(
            pairwise_scores,
            moverscore_scores.to(metric_moverscore.device),
        )
