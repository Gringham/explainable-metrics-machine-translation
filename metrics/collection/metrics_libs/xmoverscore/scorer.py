from transformers import GPT2LMHeadModel, GPT2Tokenizer, XLMRobertaConfig, XLMRobertaTokenizer, XLMRobertaModel
from transformers import BertModel, BertTokenizer, BertConfig
from metrics.collection.metrics_libs.xmoverscore.score_utils import word_mover_score, lm_perplexity

class XMOVERScorer:

    def __init__(
            self,
            model_name=None,
            lm_name=None,
            do_lower_case=False,
            device='cuda:0',
            drop_punctuation=True
    ):
        config = BertConfig.from_pretrained(model_name, output_hidden_states=True, output_attentions=True)
        self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)
        self.model = BertModel.from_pretrained(model_name, config=config)
        self.model.to(device)

        self.lm = GPT2LMHeadModel.from_pretrained(lm_name)
        self.lm_tokenizer = GPT2Tokenizer.from_pretrained(lm_name)
        self.lm.to(device)
        self.drop_punctuation = drop_punctuation

    def compute_xmoverscore(self, mapping, projection, bias, source, translations, ngram, bs, layer=None):
        score, hyp_relevance, src_relevance = word_mover_score(mapping, projection, bias, self.model, self.tokenizer, source, translations,
                                n_gram=ngram, batch_size=bs, drop_punctuation=self.drop_punctuation)
        return score, hyp_relevance, src_relevance

    def compute_perplexity(self, translations, bs):
        return lm_perplexity(self.lm, translations, self.lm_tokenizer, batch_size=bs)
