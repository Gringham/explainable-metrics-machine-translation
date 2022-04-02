import pandas as pd
import tqdm
import os
from project_root import ROOT_DIR

class WMTLoader:
    def __init__(self, lp_list=None, corpus_name='WMT17', load_from_file=None, outputFile="pandas_corpora/wmt17.tsv",
                 relative_corpus_location="./"):
        self.corpus = self.find_corpus(corpus_name)
        self.corpus_name = corpus_name
        self.relative_corpus_location = relative_corpus_location
        self.corpus_df = [[], [], [], [], [], [], []]
        self.lp_list = lp_list
        if load_from_file:
            # if the corpus was already buildt for pandas, we can load it from there
            self.corpus_df = pd.read_csv(load_from_file, delimiter='\t')
        else:
            for ref_path, lp in tqdm.tqdm(self.corpus.items()):
                if lp_list == None or lp in lp_list:
                    # load human references
                    src, tgt = lp.split('-')


                    if corpus_name != 'wmt20': # slightly different file names
                        src_path = ref_path.replace('ref', 'src').split('.')[0] + '.' + src
                        base_mt_path = os.path.join(relative_corpus_location, self.corpus_name, 'system-outputs', ref_path.split('-')[0], lp)
                    else:
                        src_path = ref_path.replace('ref', 'src').split('.')[0] + '.' + src + '.txt'
                        base_mt_path = os.path.join(relative_corpus_location, self.corpus_name, 'system-outputs', lp)

                    ref_sents = self.load_data(
                            os.path.join(relative_corpus_location, self.corpus_name, 'references', ref_path))
                    src_sents = self.load_data(
                        os.path.join(relative_corpus_location, self.corpus_name, 'sources', src_path))
                    assert len(ref_sents) == len(src_sents)

                    sp = src_path.split('-')[0]

                    # Generate lists which will be turned into a pandas dataframe
                    for mt_path in os.listdir(base_mt_path):
                        self.corpus_df[0] += [lp] * len(src_sents)
                        self.corpus_df[1] += src_sents
                        self.corpus_df[2] += ref_sents
                        mt_sents = self.load_data(base_mt_path + "/" + mt_path)

                        assert len(mt_sents) == len(src_sents)
                        self.corpus_df[3] += mt_sents
                        if corpus_name != 'wmt20':
                            # To be able to join the different system names based on name, we replace all underscore with -
                            self.corpus_df[4] += [mt_path.split('.', 1)[1].rsplit('.', 1)[0].replace('_','-')] * len(src_sents)
                            self.corpus_df[5] += [x + 1 for x in range(len(src_sents))]
                        else:
                            # In this case we need to remove the .txt
                            self.corpus_df[4] += [mt_path.split('.', 2)[2].rsplit('.', 1)[0].replace('_','-')] * len(src_sents)
                            self.corpus_df[5] += [x for x in range(len(src_sents))]

                        self.corpus_df[6] += [sp] * len(src_sents)

            self.corpus_df = pd.DataFrame(self.corpus_df,
                                          index=['LP', 'SRC', 'REF', 'HYP', 'SYSTEM', 'SID', 'TESTSET']).T

            # Allow empty hypotheses
            self.corpus_df['HYP'].fillna('', inplace=True)
            self.corpus_df.to_csv(outputFile, sep='\t')

        self.lp_list = self.corpus_df['LP'].unique().tolist()
        print("Loaded corpus:\n", self.corpus_df)

    def get_segment_da(self):
        # Only for WMT 17 - Merges sentences with DA annotations based on LP, SID and System
        seg_scores = pd.read_csv(os.path.join(self.relative_corpus_location, self.corpus_name, 'DA-seglevel.csv'),
                                 delimiter=' ')
        seg_scores = seg_scores['SYSTEM'].str.replace('_', '-')

        seg_scores = seg_scores[seg_scores['LP'].isin(self.lp_list)]

        # Only use the first system when separated by +
        seg_scores['SYSTEM'] = seg_scores.apply(lambda x: x['SYSTEM'].split('+')[0], axis=1)

        # Merge by sid
        seg_scores = pd.merge(seg_scores, self.corpus_df, how='left', left_on=['LP', 'SID', 'SYSTEM'],
                              right_on=['LP', 'SID', 'SYSTEM'])

        return seg_scores

    def get_segment_rr(self):
        # For ranking based evaluation - merge with all better and worse sentences based on lp, system and sid
        if self.corpus_name == 'wmt20':
            da_file = 'DArr-seglevel.csv'
        else:
            da_file = 'RR-seglevel.csv'

        human = pd.read_csv(os.path.join(self.relative_corpus_location, self.corpus_name, da_file),
                            delimiter=' ')
        human['BETTER'] = human['BETTER'].str.replace('_', '-')
        human['WORSE'] = human['WORSE'].str.replace('_', '-')
        human = human[human['LP'].isin(self.lp_list)]

        if self.corpus_name == 'wmt19':
            human.loc[human['LP'] == 'zh-en', "BETTER"] = human.loc[human['LP'] == 'zh-en']["BETTER"].str.rsplit('.', 1).str[0]
            human.loc[human['LP'] == 'zh-en', "WORSE"] = human.loc[human['LP'] == 'zh-en']["WORSE"].str.rsplit('.', 1).str[0]

        if self.corpus_name == 'wmt20':
            human['SID'] = human['SID'].astype(int)

        corpus_columns = self.corpus_df.columns.tolist()

        # Everything coming from the better sentences will be postpended with _better. Else with _worse
        human = pd.merge(human, self.corpus_df, how='left', left_on=['LP', 'SID', 'BETTER'],
                         right_on=['LP', 'SID', 'SYSTEM'])
        human = human.rename({col: col + '_better' for col in corpus_columns if col not in ['LP', 'SID', 'BETTER','LP', 'SRC', 'REF']}, axis=1)
        human = human.drop(columns=['SRC', 'REF'])
        human = pd.merge(human, self.corpus_df, how='left', left_on=['LP', 'SID', 'WORSE'],
                         right_on=['LP', 'SID', 'SYSTEM'])
        human = human.rename({col: col + '_worse' for col in corpus_columns if col not in ['LP', 'SID', 'BETTER','LP', 'SRC', 'REF']}, axis=1)

        return human



    def find_corpus(self, name):
        # Copied from the moverscore repo https://github.com/AIPHES/emnlp19-moverscore
        WMT2017 = dict({
            "newstest2017-csen-ref.en": "cs-en",
            "newstest2017-deen-ref.en": "de-en",
            "newstest2017-fien-ref.en": "fi-en",
            "newstest2017-lven-ref.en": "lv-en",
            "newstest2017-ruen-ref.en": "ru-en",
            "newstest2017-tren-ref.en": "tr-en",
            "newstest2017-zhen-ref.en": "zh-en"
        })

        WMT2018 = dict({
            "newstest2018-csen-ref.en": "cs-en",
            "newstest2018-deen-ref.en": "de-en",
            "newstest2018-eten-ref.en": "et-en",
            "newstest2018-fien-ref.en": "fi-en",
            "newstest2018-ruen-ref.en": "ru-en",
            "newstest2018-tren-ref.en": "tr-en",
            "newstest2018-zhen-ref.en": "zh-en",
        })

        WMT2019 = dict({
            "newstest2019-deen-ref.en": "de-en",
            "newstest2019-fien-ref.en": "fi-en",
            "newstest2019-guen-ref.en": "gu-en",
            "newstest2019-kken-ref.en": "kk-en",
            "newstest2019-lten-ref.en": "lt-en",
            "newstest2019-ruen-ref.en": "ru-en",
            "newstest2019-zhen-ref.en": "zh-en",
        })

        WMT2020 = dict({ # To en lps from wmt 20
            "newstest2020-" + lp.replace('-', '') + "-ref.en.txt": lp for lp in
            ['cs-en', 'de-en', 'iu-en', 'ja-en', 'km-en', 'pl-en', 'ps-en', 'ru-en', 'ta-en', 'zh-en']
        })

        # WMT2020 = {f : f.split("-")[1][:2] +'-'+ f.split("-")[1][2:] for f in os.listdir(os.path.join(ROOT_DIR,'metrics/corpora/WMT20/references'))}

        if name == 'wmt17':
            dataset = WMT2017
        if name == 'wmt18':
            dataset = WMT2018
        if name == 'wmt19':
            dataset = WMT2019
        if name == 'wmt20':
            dataset = WMT2020
        return dataset

    def load_data(self, path): # Also from https://github.com/AIPHES/emnlp19-moverscore/search?q=load_data
        lines = []
        # if "newstestP2020" in path and "src" in path:
        #    path = path.replace('newstestP2020','newstest2020')
        with open(path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                l = line.strip()
                lines.append(l)
        return lines


if __name__ == '__main__':
    #wmt17 = WMTLoader(corpus_name='wmt17', load_from_file=None, outputFile="pandas_corpora/wmt/full_eval/wmt17.tsv",
    #                relative_corpus_location="wmt")
    #wmt18 = WMTLoader(corpus_name='wmt18', load_from_file=None, outputFile="pandas_corpora/wmt/full_eval/wmt18.tsv",
    #                relative_corpus_location="wmt")
    #wmt19 = WMTLoader(corpus_name='wmt19', load_from_file=None, outputFile="pandas_corpora/wmt/full_eval/wmt19.tsv",
    #                relative_corpus_location="wmt")
    wmt20 = WMTLoader(corpus_name='wmt20', load_from_file=None, outputFile="pandas_corpora/wmt/full_eval/wmt20.tsv",
                    relative_corpus_location="wmt")

    #wmt17.get_segment_da().to_csv('pandas_corpora/wmt/seg_eval/wmt17_seg.tsv', sep='\t')
    #wmt18.get_segment_rr().to_csv('pandas_corpora/wmt/seg_eval/wmt18_seg.tsv', sep='\t')
    #wmt19.get_segment_rr().to_csv('pandas_corpora/wmt/seg_eval/wmt19_seg.tsv', sep='\t')
    wmt20.get_segment_rr().to_csv('pandas_corpora/wmt/seg_eval/wmt20_seg.tsv', sep='\t')

