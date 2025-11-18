from __future__ import annotations

from dataclasses import dataclass
from mbrs.metrics.base import MetricReferenceless, register
from tqdm import tqdm
import torch
import numpy as np

from ..modules import FENICE

@register("fenice")
class MetricFENICE(MetricReferenceless):
    """
    Metric class for FENICE reference-free metric.
    """

    HIGHER_IS_BETTER: bool = True

    @dataclass
    class Config(MetricReferenceless.Config):
        batch_size: int = 8
        backup_batch_size: int = max(1, batch_size//4)
        claim_extractor_batch_size: int = batch_size*4
        nli_batch_size: int = batch_size*4
        use_coref: bool = False
        num_sent_per_paragraph: int = 5
        sliding_paragraphs: bool = True
        sliding_stride: int = 1
        doc_level_nli: bool = True
        paragraph_level_nli: bool = True
        coreference_batch_size: int = 1
        nli_max_length: int = 1024

    cfg: MetricFENICE.Config

    def __init__(self, cfg: MetricFENICE.Config):
        super().__init__(cfg)
        self.scorer : FENICE = FENICE(
            use_coref=cfg.use_coref,
            num_sent_per_paragraph=cfg.num_sent_per_paragraph,
            sliding_paragraphs=cfg.sliding_paragraphs,
            sliding_stride=cfg.sliding_stride,
            doc_level_nli=cfg.doc_level_nli,
            paragraph_level_nli=cfg.paragraph_level_nli,
            claim_extractor_batch_size=cfg.claim_extractor_batch_size,
            coreference_batch_size=cfg.coreference_batch_size,
            nli_batch_size=cfg.nli_batch_size,
            nli_max_length=cfg.nli_max_length,
        )
        self.backup_scorer : FENICE = FENICE(
            use_coref=cfg.use_coref,
            num_sent_per_paragraph=cfg.num_sent_per_paragraph,
            sliding_paragraphs=cfg.sliding_paragraphs,
            sliding_stride=cfg.sliding_stride,
            doc_level_nli=cfg.doc_level_nli,
            paragraph_level_nli=cfg.paragraph_level_nli,
            claim_extractor_batch_size=max(1, cfg.claim_extractor_batch_size//4),
            coreference_batch_size=cfg.coreference_batch_size,
            nli_batch_size=max(1, cfg.nli_batch_size//4),
            nli_max_length=cfg.nli_max_length,
        )

    def scores(
        self, hypotheses: list[str], sources: list[str]
    ) -> dict:
        temp_scores: list[dict] = []
        data = [{"document": src, "summary": hyp} for src, hyp in zip(sources, hypotheses)]
        for i in tqdm(
            range(0, len(data), self.cfg.batch_size), 
            desc="Calculating FENICE scores", 
            total=len(data)//self.cfg.batch_size
        ):
            batch_data = data[i:i+self.cfg.batch_size]
            try:
                res = self.scorer.score_batch(batch_data)
            except RuntimeError as e:
                print(f"Runtime error during FENICE scoring: {e}")
                if 'out of memory' in str(e):
                    print("Out of memory error, using backup scorer with smaller batch size")
                    torch.cuda.empty_cache()
                    for j in tqdm(
                        range(0, len(batch_data), self.cfg.backup_batch_size), 
                        desc="Calculating FENICE scores with backup scorer", 
                        total=len(batch_data)//self.cfg.backup_batch_size
                    ):
                        sub_batch_data = batch_data[j:j+self.cfg.backup_batch_size]
                        sub_res = self.backup_scorer.score_batch(sub_batch_data)
                        temp_scores.extend(sub_res)
                else:
                    raise e
            else:
                temp_scores.extend(res)
        
        assert len(temp_scores) == len(hypotheses), \
            f"Length of FENICE scores {len(temp_scores)} does not match number of hypotheses {len(hypotheses)}"
        
        scores = {
            "score": [v["score"] for v in temp_scores],
            "alignments": [v["alignments"] for v in temp_scores]
        }            
        return scores

    def score(
        self, hypothesis: str, source: str
    ) -> dict:
        score = self.scorer.score_batch(
            [{"document": source, "summary": hypothesis}]
        )[0]
        return score

    def corpus_score(self, hypotheses: list[str], sources: list[str]) -> float:
        scores = self.scores(hypotheses, sources)
        fenice_scores = scores["score"]
        return float(np.array(fenice_scores).mean())