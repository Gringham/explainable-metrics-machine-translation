import pandas as pd
import os

from scipy.stats import pearsonr

from project_root import ROOT_DIR


def evaluate_seg(scores_df, metrics, output = os.path.join(ROOT_DIR,'metrics/outputs/test_file.tsv'), da_column='HUMAN'):
    corpus = scores_df.drop(columns=['SRC', 'REF', 'HYP'])

    lp_list = corpus['LP'].unique().tolist()

    # Calculate pearson correlation for the specified metrics
    csv = 'LP\t' + '\t'.join(metrics) + '\n'
    for lp in lp_list:
        csv += lp
        for metric in metrics:
            preds = corpus.loc[corpus['LP'] == lp][da_column]
            labels = corpus.loc[corpus['LP'] == lp][metric]
            print(preds[0],labels[0])
            csv += "\t" + str(pearsonr(preds, labels)[0])
        csv += '\n'

    if output:
        with open(output, 'w') as f:
            f.write(csv)

    return csv

if __name__ == '__main__':
    metrics = ['TRANSQUEST']
    #metrics = ['BLEU','PSEUDO1', 'PSEUDO2', 'PSEUDO3', 'SACREBLEU','METEOR','COMET', 'TRANSQUEST','BERTSCORE','MOVERSCORE','xmoverscore_clp1','xmoverscore_clp2','xmoverscore_umd1','xmoverscore_umd2','LABSE','XLMR','xmoverscore_clp1_lm','xmoverscore_clp2_lm','xmoverscore_umd1_lm','xmoverscore_umd2_lm']
    scores = pd.read_csv(os.path.join(ROOT_DIR,'metrics/outputs/WMT_17_SEG_SCORES_RAW3'), delimiter='\t')
    evaluate_seg(scores, metrics)