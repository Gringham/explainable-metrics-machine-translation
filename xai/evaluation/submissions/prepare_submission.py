import json, string
import pandas as pd
import statistics
from project_root import ROOT_DIR
import os

def align_scores(ck, h):
    # This function is necessary to align the feature importance scores in case subword tokenization.
    # There might be some failure cases to this function
    res = []
    ck_pos = 0
    hyp_pos = 0
    hyp = h.lower().split()
    ck = [(c[0], c[1].lower()) for c in ck]

    # separate comma - necessary to prevent some failure cases of alignment (in case of xmoverscore with dropped punct.)
    # It will copy the score of the previous token for punctutation:
    ck2 = []
    for c in ck:
        if len(c[1])>0:
            if (c[1][-1] ==',' or c[1][-1] =='.' or c[1][-1] =='(' or c[1][-1] ==')') and len(c[1])>1:
                ck2.append((c[0],c[1][:-1]))
                ck2.append((c[0],c[1][-1]))
            else:
                ck2.append(c)
    ck = ck2

    # Go through all positions in the hypothesis and align token-level scores by averaging
    while hyp_pos < len(hyp):
        if hyp[hyp_pos] == ck[ck_pos][1]:
            # if both are equal for a token, no alignment is necessary
            res.append(ck[ck_pos])

        elif ck[ck_pos][1] in hyp[hyp_pos] or '[unk]' in ck[ck_pos][1] or '[UNK]' in ck[ck_pos][1]:
            # if the assigned scores token is part of the hypothesis token, we have a look at the next tokens in assigned scores
            int_string = ck[ck_pos][1]
            int_scores = [ck[ck_pos][0]]

            # hyp without punctuation to capture some cases
            comp = hyp[hyp_pos].translate(str.maketrans('', '', string.punctuation))

            if  ck_pos < len(ck)- 2 :
                while int_string + ck[ck_pos + 1][1] in hyp[hyp_pos] or int_string + ck[ck_pos + 1][1] in comp :
                    # check, whether the current and next token together are still a part of the next word
                    # if yes, collect the current scores and add the current string
                    ck_pos += 1
                    int_string += ck[ck_pos][1]
                    int_scores.append(ck[ck_pos][0])
                    if ck_pos + 1 == len(ck):
                        break

            # Average over all tokens that belong to a word
            res.append((sum(int_scores) / len(int_scores), int_string))

        elif hyp[hyp_pos] in ck[ck_pos][1]:
            # if the hypothesis token is part of the assigned scores token, we have a look at the next tokens in the hypothesis
            int_string = hyp[hyp_pos]
            res.append((ck[ck_pos][0], hyp[hyp_pos]))

            while int_string + hyp[hyp_pos + 1] in ck[ck_pos][1]:
                # As long as multiple words in the hyp are part of one token in the importance scores, we add the words
                # in the hypothesis with the score of that token
                hyp_pos += 1
                int_string += hyp[hyp_pos]
                res.append((ck[ck_pos][0], hyp[hyp_pos]))
                if hyp_pos+1 ==  len(hyp):
                    break

        else:
            # in all other cases we assume that tokenization has dropped this token and assign it with the median of scores
            res.append((statistics.median([c[0] for c in ck]),hyp[hyp_pos]))

            # don't change the ck position in this case
            ck_pos -= 1

        ck_pos += 1
        hyp_pos += 1

        # if there are tokens at the end that were deleted by the tool
        if len(ck) == ck_pos:
            while hyp_pos < len(hyp):
                res.append((statistics.median([c[0] for c in ck]), hyp[hyp_pos]))
                hyp_pos += 1

    return res

def extract_attributions(df, explanations, attribution_key = 'attributions', comp = 'HYP', invert = True):
    # load explanations
    attributions_per_key = {
        key: [[(w[0], w[1]) if type(w[1]) == str else (w[1], w[0]) for w in
               explanations[x]['metrics'][key][attribution_key]] for x in
              range(len(explanations))] for key in explanations[0]['metrics']}

    unaligned = []
    # Check where explanations are unaligned with the original hypothesis
    for key in attributions_per_key:
        for x in range(len(attributions_per_key[key])):
            if len(attributions_per_key[key][x]) != len(df[comp][x].split(' ')):
                unaligned.append(x)
                #print("Problem detected:",x, attributions_per_key[key][x], df[comp][x])

    # Fix these cases using an alignment function
    for key in attributions_per_key:
        for x in unaligned:
            #print("Trying to fix", x, 'by averaging')
            attributions_per_key[key][x] = align_scores(attributions_per_key[key][x], df[comp][x])

    # Check if alignment fixed the problem
    for key in attributions_per_key:
        for x in range(len(attributions_per_key[key])):
            assert len(attributions_per_key[key][x]) == len(df[comp][x].split(' '))

    # drop words
    attributions_per_key = {key: [[word[0] for word in sent] for sent in value] for key, value in
                            attributions_per_key.items()}

    # there is a negative correlation of attributions and errors
    if invert:
        for key in attributions_per_key.keys():
            attributions_per_key[key] = [[-b for b in a] for a in attributions_per_key[key]]

    return attributions_per_key


if __name__ == '__main__':
    # Change all lp occurances if you want to use it for another language

    path = os.path.join(ROOT_DIR,'xai/output/explanations/0_999_mlqe_de_dev_attributions_bs.json')
    invert = True
    lp = 'et_en'
    ret_key = 'BERTSCORE'
    # check we are matching the structure of the orig file
    test_df = pd.read_csv(os.path.join(ROOT_DIR,'metrics/corpora/pandas_corpora/eval4NLP/eval4nlp_dev_' + lp + '.tsv'), delimiter='\t')

    with open(path, 'r') as f:
        explanations = json.loads(f.read())

    # load all explanations in the submission format of the shared task
    attributions_per_key = extract_attributions(test_df, explanations)
    src_attributions_per_key = extract_attributions(test_df, explanations, attribution_key='src_attributions', comp='SRC')

    with open('target.submission', 'w') as f:
        for item in attributions_per_key[ret_key]:
            f.write(' '.join([str(i) for i in item])+"\n")

    with open('source.submission', 'w') as f:
        for item in src_attributions_per_key[ret_key]:
            f.write(' '.join([str(i) for i in item])+"\n")

    with open('sentence.submission', 'w') as f:
        for x in range(len(explanations)):
            f.write(str(explanations[x]['metrics'][ret_key]['score'])+"\n")