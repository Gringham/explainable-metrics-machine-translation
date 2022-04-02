from pytorch_transformers import AutoConfig, AutoTokenizer, AutoModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer, XLMRobertaConfig, XLMRobertaTokenizer, XLMRobertaModel
from transformers import BertModel, BertTokenizer, BertConfig
from metrics.collection.metrics_libs.xmoverscore_xlmr.score_utils import word_mover_score, lm_perplexity
#from metrics.collection.metrics_libs.xmoverscore_xlmr.score_utils_v2 import word_mover_score, lm_perplexity
#from metrics.collection.metrics_libs.xmoverscore_xlmr.score_utils_summarized_dist_matrix import word_mover_score, lm_perplexity



class XMOVERScorer:

    def __init__(
            self,
            model_name=None,
            lm_name=None,
            do_lower_case=False,
            device='cuda:0',
            drop_punctuation=None
    ):
        #config = BertConfig.from_pretrained(model_name, output_hidden_states=True, output_attentions=True)
        #self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)
        #self.model = BertModel.from_pretrained(model_name, config=config)
        #self.model.to(device)

        # Using xlmr instead of bert
        config = XLMRobertaConfig.from_pretrained(model_name, output_hidden_states=True, output_attentions=True)
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
        self.model = XLMRobertaModel.from_pretrained(model_name, config=config)
        self.model.to(device)

        self.lm = GPT2LMHeadModel.from_pretrained(lm_name)
        self.lm_tokenizer = GPT2Tokenizer.from_pretrained(lm_name)
        self.lm.to(device)

    def compute_xmoverscore(self, mapping, projection, bias, source, translations, ngram, bs, layer=21):
        # additionally returning the importance scores
        score, hyp_relevance, src_relevance = word_mover_score(mapping, projection, bias, self.model, self.tokenizer, source, translations,
                                n_gram=ngram, batch_size=bs, layer = layer)
        return score, hyp_relevance, src_relevance

    def compute_perplexity(self, translations, bs):
        return lm_perplexity(self.lm, translations, self.lm_tokenizer, batch_size=bs)
