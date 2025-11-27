from .moverscore import word_mover_score, get_idf_dict

class MoverScorer:
    def __init__(self, stop_words=[], n_gram=1, remove_subwords=True, batch_size=256, device='cuda'):
        self.device = device
        self.stop_words = stop_words
        self.n_gram = n_gram
        self.remove_subwords = remove_subwords
        self.batch_size = batch_size

    def score(self, references, translations):
        idf_dict_ref = get_idf_dict(references)
        idf_dict_hyp = get_idf_dict(translations)
        
        scores = word_mover_score(
            references, 
            translations, 
            idf_dict_ref, 
            idf_dict_hyp, 
            stop_words=self.stop_words, 
            n_gram=self.n_gram, 
            remove_subwords=self.remove_subwords, 
            batch_size=self.batch_size,
            device=self.device
        )
        return scores