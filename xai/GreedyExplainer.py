import random
import tqdm
import torch
import numpy as np
from more_itertools import chunked
from pytorch_transformers import AutoTokenizer, AutoModelWithLMHead

from metrics.collection.Bleurt import Bleurt


class MaskAttack:
    '''
    A feature importance explainer and class to find inverse metrics using an algorithm that finds perturbations that
    receive a score close to a target score
    '''

    def __init__(self, model_name='bert-base-cased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelWithLMHead.from_pretrained(model_name)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def n_max(self, l, n):
        # Returns the nth largest element from a list
        l_set = set(l)
        for x in range(n):
            if len(l_set)==1:
                break
            l_set.remove(max(l_set))
        return (max(l_set))


    def mask_bert_sim(self, sents=["My dog is hungry.", "I am tired."], masks=[[0, 1, 0, 1], [2, 0, 0]]):
        '''
        :param sents: Sentences to permute
        :param masks: Masks, where each position indicates the nth likeliest word to replace each word with
        :return: permuted sentences
        '''

        # Get positions of masked elements
        mask_ids = [[i for i, x in enumerate(mask) if x != 0] for mask in masks]
        [random.shuffle(m) for m in mask_ids]
        max_mask = max([len(mask_id) for mask_id in mask_ids])

        modified_sents = None

        # replace one mask at a time
        for x in range(max_mask):
            masked_tokens = []

            # after the first permutation continue with the permuted sentences
            if modified_sents:
                sents = modified_sents

            # Mask out the next position
            for y in range(len(mask_ids)):
                tokens = sents[y].split(" ")
                if (x < len(mask_ids[y])):
                    masked_tokens.append(tokens[:mask_ids[y][x]] + ['[MASK]'] + tokens[mask_ids[y][x] + 1:])
                else:
                    masked_tokens.append(tokens)

            masked_tokens = [self.tokenizer.tokenize(' '.join(['[CLS]'] + t + ['[SEP]'])) for t in masked_tokens]

            # Generates segment ids based on the token lengths an determines max len for padding
            max_len = max(len(t) for t in masked_tokens)
            seg_ids = [[0] * len(i) for i in masked_tokens]

            # Generating the padding with a different id
            for pad in range(len(masked_tokens)):
                while len(masked_tokens[pad]) < max_len:
                    masked_tokens[pad].append('[PAD]')
                    seg_ids[pad].append(1)

            # Getting the index after re-tokenization
            mask_indices = [t.index('[MASK]') if '[MASK]' in t else -1 for t in masked_tokens]

            # Getting the ids for every token
            ids = [self.tokenizer.convert_tokens_to_ids(t) for t in masked_tokens]

            # Move ids to tensor
            id_tensor = torch.tensor(ids, device=self.device)
            seg_id_tensor = torch.tensor(seg_ids, device=self.device)

            # Get Predictions
            self.model.eval()

            bert_scores = []

            with torch.no_grad():
                for batch in chunked(range(len(id_tensor)), 64):
                    id_in_batch = id_tensor[batch]
                    seg_id_in_batch = seg_id_tensor[batch]
                    predictions = self.model(id_in_batch, seg_id_in_batch)
                    del id_in_batch, seg_id_in_batch

                    # Get the predictions for the masked token
                    bert_scores += [predictions[0][z, mask_indices[z]].tolist() if mask_indices[z] != -1 else [] for z in
                                    range(predictions[0].shape[0])]

            # Get token corresponding to indices
            indices = [[x for x in range(0, len(b))] for b in bert_scores]

            # get the n-th best score for every masked value in the input sentences
            scoring = [indices[z][bert_scores[z].index(self.n_max(bert_scores[z],masks[z][mask_ids[z][x]]-1))] if len(bert_scores[z]) > 0 else -1 for z
                       in
                       range(len(bert_scores))]

            sents = [' '.join(s) for s in masked_tokens]
            # Clean up the modified sentences
            modified_sents = [s.replace('[CLS]', '').replace('[PAD]', '').replace('[SEP]', '').strip() for s in sents]
            modified_sents = [
                modified_sents[m].replace('[MASK]', self.tokenizer.convert_ids_to_tokens(scoring[m])) if scoring != -1 else
                modified_sents[m] for m in range(len(modified_sents))]

            del predictions, bert_scores, indices, scoring, sents

        return modified_sents

    def gen_masks(self, base, p=0.1):
        '''
        :param base: Increments base masks with other masks
        :param p: Probability for increments
        :return:
        '''
        return [base[i]+np.random.choice(2,len(base[i]),p=[1-p,p]) for i in range(len(base))]

    def attack(self, hyp, metric, tgt_score = 1, permutations_per_step = 20, steps = 10, p = 0.1):
        '''
        This can be considered as a targeted unconstrained attack
        :param hyp: hypothesis sentence
        :param metric: A metric to attack
        :param tgt_score: Target score
        :param permutations_per_step: Number of masks to try in each step
        :param steps: Number of incrementation steps
        :param p: Permutation probability in this incrementation step
        :return: permuted sentence, score, mask, original score
        '''
        current_mask = None
        orig = metric([hyp])

        # Initialize empty masls
        base = [[0] * len(hyp.split())] * permutations_per_step

        # initialize the update variables
        best_score = orig[0]
        best_diff = np.abs(best_score - tgt_score)
        best_mask = base[0]
        best_sentence = hyp

        for x in tqdm.tqdm(range(steps), desc="Step"):
            # For each step generate masks, permute the sentences and determine the scores
            if x == 0:
                masks = self.gen_masks(base, p)
            else:
                # new masks are generated based on the best mask of the last iteration
                masks = self.gen_masks([current_mask] * permutations_per_step, p)
            sents = self.mask_bert_sim([hyp] * permutations_per_step, masks)
            scores = metric(sents)

            # Find the score closest to the target and update the values accordingly.
            target_diff = [np.abs(score - tgt_score) for score in scores]
            id = np.argmin(target_diff)
            current_diff = target_diff[id]
            current_mask = masks[id]

            if current_diff < best_diff:
                best_sentence = sents[id]
                best_score = scores[id]
                best_mask = masks[id]
                best_diff = current_diff
            #print(current_diff, masks)

        return best_sentence, best_score, best_mask, orig

    def explain(self, hyp, metric):
        '''
        Here we use the lowest and highes reachable target scores starting from a hypothesis and use the mask values as
        feature importance scores
        :param hyp: hypothesis sentences
        :param metric: a metric to explain
        :return: feature importance tuples
        '''
        _, _, worst_mask, _ = self.attack(hyp, metric, -1000, 30, 5, 0.15)
        _, _, best_mask, orig = self.attack(hyp, metric, 1000, 30, 5, 0.15)
        return ([('dummy',(-b + w).item()) for b, w in zip(best_mask, worst_mask)], orig)

    def apply_explanation(self, df, metrics=None, recover=False):
        '''
        :param df: A dataframe to explain
        :param metrics: A list of metric names to explain. When none, all metrics known by MetricWrapper will be explained.
        :param recover: Whether to recover previous explanations
        :return: A dictionary with feature importance explanations
        '''
        explanations = self.MW.apply_hyp_explainer(df,
                                                   self.explain,
                                                   metrics=metrics,
                                                   recover=recover)

        # Unpacking the explanation object
        attributions = []
        for x in range(len(explanations)):
            print('Sample:', explanations[x])
            attribution = {'src': explanations[x]['src'],
                           'ref': explanations[x]['ref'],
                           'hyp': explanations[x]['hyp'],
                           'metrics': {key: {'attributions': value[0],
                                             'score': value[1]
                                             }
                                       for key, value in explanations[x]['metrics'].items()},
                           'times': {key: {'time': value}
                                     for key, value in explanations[x]['times'].items()}
                           }

            print(attribution)
            attributions.append(attribution)
        return attributions

if __name__ == '__main__':
    import os
    os.environ["HOME"] = "."
    MA = MaskAttack()
    import tensorflow as tf

    # Copied from https://www.tensorflow.org/guide/gpu
    # To reduce memory used by tensorflow, e.g. for BERT-attack
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

    # Find neighborhood sentences for BLEURT
    metric_base = Bleurt()

    ref = 'My cat is old.'
    hyp = 'My cat lives since 17 years.'
    src = 'Ich habe einen Hund.'
    res = []

    metric = metric_base.get_abstraction(ref)

    for x in np.arange(-1, 1, 0.2):
        res.append(MA.attack(hyp, metric, x, 20, 20, 0.1))


    print("REF:", ref, "HYP:", hyp, "Original Score:", metric([hyp])[0])
    for r in res:
        print("HYP*:", r[0], "Pert. Score", round(r[1], 3))

    '''
    # Using it to create feature importance scores
    explain_corpus(SE,
                   recover=False,
                   metrics=['TRANSQUEST'],
                   from_row=0, to_row=100,
                   outfile='et_en_test_attributions_gen_mask_tq'),
                   mlqe_pandas_path=os.path.join(ROOT_DIR,'metrics/corpora/pandas_corpora/eval4nlp_test_et-en.tsv'))
    evaluate_mlqe_auc([
        os.path.join(ROOT_DIR,'xai/output/explanations/0_99_et_en_test_attributions_gen_mask_tq')],
        start=0, end=100, invert=True, f=100)
    '''