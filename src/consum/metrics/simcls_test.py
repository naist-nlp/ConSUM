import pytest
import torch

from .simcls import MetricSimCLS

from SimCLS.model import ReRanker
from SimCLS.data_utils import ReRankingDataset, collate_mp, to_cuda
from torch.utils.data import DataLoader
from functools import partial

from ..utils.summ_test_set import SIMCLS_DATA_PATH as DATA_PATH, SIMCLS_MODEL_PATH as MODEL_PATH, SIMCLS_SCORES


class TestMetricSimCLS:
    @pytest.fixture(scope="class")
    def metric_simcls(self):
        return MetricSimCLS(MetricSimCLS.Config(simcls_path=MODEL_PATH, batch_size=4))

    def test_score(self, metric_simcls: MetricSimCLS):
        try:
            metric_simcls.score("hyp", "src")
        except NotImplementedError:
            pass
        
    def test_scores(self, metric_simcls: MetricSimCLS):
        scores = metric_simcls.scores(
            data_path=DATA_PATH,
            candidate_cnt=4,
        )

        simcls_scores = [d["simcls"] for d in SIMCLS_SCORES]
        
        torch.testing.assert_close(
            torch.Tensor(scores),
            torch.Tensor(simcls_scores),
        )