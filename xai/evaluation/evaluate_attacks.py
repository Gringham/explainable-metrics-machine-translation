import os

import dill
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import language_tool_python
import tqdm
from sentence_transformers.util import pytorch_cos_sim
from sentence_transformers import SentenceTransformer
from project_root import ROOT_DIR


# Here we use language tool and sentence similarity to grade adversarial attacks, which was proposed to do in:
'''
John Morris, Eli Lifland, Jack Lanchantin, Yangfeng Ji, and Yanjun Qi. “Reevaluating Adversarial
Examples in Natural Language”. In: Findings of the Association for Computational Linguistics: EMNLP
2020. Online: Association for Computational Linguistics, Nov. 2020, pp. 3829–3839. doi: 10.18653/
v1/2020.findings-emnlp.341. url: https://aclanthology.org/2020.findings-
emnlp.341.'''
tool = language_tool_python.LanguageTool('en-US')
model = SentenceTransformer('paraphrase-MiniLM-L12-v2')



# Specify the paths to objects that were returned by the different attackers
paths = {
    'bert-attack': os.path.join(ROOT_DIR,'xai/output/explanation_checkpoints/adv_attack_de_en_2_to_0_bertattack_wmt19.dill'),
    'textfooler': os.path.join(ROOT_DIR,'xai/output/explanation_checkpoints/adv_attack_de_en_2_to_0_textfooler_wmt19.dill'),
    'textfooler_adjusted': os.path.join(ROOT_DIR,'xai/output/explanation_checkpoints/adv_attack_de_en_2_to_0_textfooler_adjusted_wmt19.dill')
}

# Loads the explanations/attacks to a dict
explanations = {}
for attack, path in paths.items():
    with open(path, 'rb') as pickle_file:
        explanations[attack] = dill.load(pickle_file)
        num_explanations = len(explanations[attack])

attacks_combined = [{key: explanations[key][x] for key in explanations} for x in range(num_explanations)]
metrics = list(attacks_combined[0][attack]['metrics'].keys())

# Depends on whether you want these two metrics included
metrics.remove('TRANSLATIONBLEU')
metrics.remove('TRANSLATIONMETEOR')

results = None
hyps = []
srcs = []
refs = []
# In a first loop we extract the relevant information for several metrics
for sample in tqdm.tqdm(attacks_combined):
    for attack in sample:
        src = sample[attack]['src']
        hyp = sample[attack]['hyp']
        ref = sample[attack]['ref']
        hyps.append(hyp)
        refs.append(ref)
        srcs.append(src)
        sample[attack]['metrics'].pop('TRANSLATIONBLEU')
        sample[attack]['metrics'].pop('TRANSLATIONMETEOR')
        sample[attack]['metrics'].pop('PSEUDO1')
        sample[attack]['metrics'].pop('PSEUDO2')
        sample[attack]['metrics'].pop('PSEUDO3')
        sample[attack]['metrics'].pop('BLEU')
        sample[attack]['metrics'].pop('XLMR')
        sample[attack]['metrics'].pop('LABSE')

        # check the original number of grammatic errors
        # COMMENT THIS IN FOR GRAMMATICAL ERROR RATE
        orig_check = tool.check(hyp)
        print(orig_check)
        orig_num_errors = len(orig_check)


        # For attacks that were computed with the textattack framework
        if attack == "bert-attack" or attack == "checklist" or attack == "textfooler" or attack == "textfooler_adjusted":
            sample_metrics = sample[attack]['metrics']


            try:
                result_df_dict = {
                    'src' :  pd.DataFrame(
                        {metric: src
                         for metric in sample_metrics}, index=[0]),
                    'hyp': pd.DataFrame(
                        {metric: hyp
                         for metric in sample_metrics}, index=[0]),
                    'ref': pd.DataFrame(
                        {metric: ref
                         for metric in sample_metrics}, index=[0]),
                    'orig_class_per_metric': pd.DataFrame(
                        {metric: sample_metrics[metric][0][attack].original_result.ground_truth_output
                         for metric in sample_metrics}, index=[0]),
                    'perturbed_class_per_metric': pd.DataFrame(
                        {metric: sample_metrics[metric][0][attack].perturbed_result.output
                         for metric in sample_metrics}, index=[0]),
                    'perturbed_text_per_metric': pd.DataFrame(
                        {metric: sample_metrics[metric][0][attack].perturbed_result.attacked_text.text
                         for metric in sample_metrics}, index=[0]),

                    # Attack success state
                    'status_per_metric': pd.DataFrame(
                        {metric: sample_metrics[metric][0][attack].perturbed_result.goal_status
                         for metric in sample_metrics}, index=[0]),

                    # The number of modified indices divided by the number of overall words
                    'perturbation_rate_per_metric': pd.DataFrame(
                        {metric: len(sample_metrics[metric][0][attack].perturbed_result.attacked_text.attack_attrs[
                                         'modified_indices']) / sample_metrics[metric][0][
                                     attack].perturbed_result.attacked_text.num_words
                         for metric in sample_metrics}, index=[0]),

                    # The sentence similarity of the hypoithesis and the adversarial
                    'semantic_errors_per_metric': pd.DataFrame(
                        {metric: pytorch_cos_sim(model.encode(hyp, convert_to_tensor=True), model.encode(
                            sample_metrics[metric][0][attack].perturbed_result.attacked_text.text,
                            convert_to_tensor=True)).cpu().item()
                         for metric in sample_metrics}, index=[0]),

                    # The number of grammatic errors in the adversarial minus the number in the original
                    'grammatical_errors_per_metric': pd.DataFrame(
                        {metric: len(tool.check(
                            sample_metrics[metric][0][attack].perturbed_result.attacked_text.text)) - orig_num_errors
                         for metric in sample_metrics}, index=[0]),
                }
            except Exception as e:
                print("Issue Skipped ..., this should be taken a look at",e, sample_metrics)

        # Add attack type and produce dataframe from dictionary
        for df in result_df_dict:
            result_df_dict[df]['attack'] = attack
        if results == None:
            results = result_df_dict
        else:
            for df in result_df_dict:
                results[df] = pd.concat([results[df], result_df_dict[df]])

with open("saved_attack_results", 'wb') as pickle_file:
    # Dill provides more options than pickle
    dill.dump(results, pickle_file, -1)


# Plot the attack results as graphs
full_drops = {}
pert_rate = {}
gram_rate = {}
sem_rate = {}
for attack in paths:
    # Filter df for attack type
    orig_per_attack = results['orig_class_per_metric'][results['orig_class_per_metric']['attack'] == attack]
    pert_per_attack = results['perturbed_class_per_metric'][results['perturbed_class_per_metric']['attack'] == attack]
    pert_rate_attack = results['perturbation_rate_per_metric'][
        results['perturbation_rate_per_metric']['attack'] == attack]
    status_per_attack = results['status_per_metric'][results['status_per_metric']['attack'] == attack]
    sem_per_attack = results['semantic_errors_per_metric'][results['semantic_errors_per_metric']['attack'] == attack]
    gram_rate_attack = results['grammatical_errors_per_metric'][ results['status_per_metric']['attack'] == attack]

    #zeroes, ones, threes = [], [], []

    full_drops[attack] = []
    pert_rate[attack] = []
    gram_rate[attack] = []
    sem_rate[attack] = []
    for metric in metrics:
        # original and changed class
        o=2
        t=0

        # Ratio of successful attacks to all attacks per metric
        full_drops[attack].append(
            len(status_per_attack[metric][(orig_per_attack[metric] == o) & (pert_per_attack[metric] == t)]) / len(
                status_per_attack[metric]))

        # Average perturbations
        sel1 = pert_rate_attack[metric][(orig_per_attack[metric] == o) & (pert_per_attack[metric] == t)]
        pert_rate[attack].append(sum(sel1) / len(sel1) if len(sel1) > 0 else -1)

        # Average grammatic error introduced
        sel2 = gram_rate_attack[metric][(orig_per_attack[metric] == o) & (pert_per_attack[metric] == t)]
        gram_rate[attack].append(sum(sel2) / len(sel2) if len(sel2) > 0 else -100)

        # Average semantic sim
        sel3 = sem_per_attack[metric][(orig_per_attack[metric] == o) & (pert_per_attack[metric] == t)]
        sem_rate[attack].append(sum(sel3) / len(sel3) if len(sel3) > 0 else -100)

sns.set(font_scale=1.5)
# Plot each of the computed dataframes with matplotlib
full_drop_df = pd.DataFrame(full_drops)
full_drop_df = full_drop_df.rename(index={x: metrics[x] for x in range(len(metrics))}).transpose()
full_drop_df = full_drop_df.sort_values(by='bert-attack', axis=1, ascending=False)
full_drop_df = full_drop_df.rename(index={"bert-attack":"BERT-Attack","textfooler":"TEXTFOOLER","textfooler_adjusted":"TFADJUSTED"})
full_drop_df = full_drop_df.rename(columns={
    "BERTSCORE":"BERTScore",
    "BLEURT":"BLEURT",
    "COMET":"COMET",
    "METEOR":"METEOR",
    "MOVERSCORE":"MoverScore",
    "SCAREBLEU":"ScareBLEU - Sentence BLEU",
    "SENTCHRF":"Sentence CHRF",
    "TRANSQUEST":"TransQuest",
    "XMOVERSCORE":"XMoverScore",

})
ax = plt.axes()
sns.heatmap(full_drop_df, annot=True)
ax.set_title('')
plt.tight_layout()
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=45, ha='right')
plt.show()

pert_rate_df = pd.DataFrame(pert_rate)
pert_rate_df = pert_rate_df.rename(index={x: metrics[x] for x in range(len(metrics))}).transpose()
pert_rate_df = pert_rate_df.sort_values(by='bert-attack', axis=1, ascending=False)
pert_rate_df = pert_rate_df.rename(index={"bert-attack":"BERT-Attack","textfooler":"TEXTFOOLER","textfooler_adjusted":"TFADJUSTED"})
pert_rate_df = pert_rate_df.rename(columns={
    "BERTSCORE":"BERTScore",
    "BLEURT":"BLEURT",
    "COMET":"COMET",
    "METEOR":"METEOR",
    "MOVERSCORE":"MoverScore",
    "SCAREBLEU":"ScareBLEU - Sentence BLEU",
    "SENTCHRF":"Sentence CHRF",
    "TRANSQUEST":"TransQuest",
    "XMOVERSCORE":"XMoverScore",

})
ax = plt.axes()
sns.heatmap(pert_rate_df, annot=True)
# ax.set_title('4->0 Perturbation Ratio of Successful Attacks')
plt.tight_layout()
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=45, ha='right')
plt.show()

gram_rate_df = pd.DataFrame(gram_rate)
gram_rate_df = gram_rate_df.rename(index={x: metrics[x] for x in range(len(metrics))}).transpose()
gram_rate_df = gram_rate_df.sort_values(by='bert-attack', axis=1, ascending=False)
gram_rate_df = gram_rate_df.rename(index={"bert-attack":"BERT-Attack","textfooler":"TEXTFOOLER","textfooler_adjusted":"TFADJUSTED"})
gram_rate_df = gram_rate_df.rename(columns={
    "BERTSCORE":"BERTScore",
    "BLEURT":"BLEURT",
    "COMET":"COMET",
    "METEOR":"METEOR",
    "MOVERSCORE":"MoverScore",
    "SCAREBLEU":"ScareBLEU - Sentence BLEU",
    "SENTCHRF":"Sentence CHRF",
    "TRANSQUEST":"TransQuest",
    "XMOVERSCORE":"XMoverScore",

})
ax = plt.axes()
sns.heatmap(gram_rate_df, annot=True)
# ax.set_title('3->0 Perturbation Ratio of Successful Attacks')
plt.tight_layout()
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=45, ha='right')
plt.show()

sem_rate_df = pd.DataFrame(sem_rate)
sem_rate_df = sem_rate_df.rename(index={x: metrics[x] for x in range(len(metrics))}).transpose()
sem_rate_df = sem_rate_df.sort_values(by='bert-attack', axis=1, ascending=False)
sem_rate_df = sem_rate_df.rename(index={"bert-attack":"BERT-Attack","textfooler":"TEXTFOOLER","textfooler_adjusted":"TFADJUSTED"})
sem_rate_df = sem_rate_df.rename(columns={
    "BERTSCORE":"BERTScore",
    "BLEURT":"BLEURT",
    "COMET":"COMET",
    "METEOR":"METEOR",
    "MOVERSCORE":"MoverScore",
    "SCAREBLEU":"ScareBLEU - Sentence BLEU",
    "SENTCHRF":"Sentence CHRF",
    "TRANSQUEST":"TransQuest",
    "XMOVERSCORE":"XMoverScore",

})
ax = plt.axes()
sns.heatmap(sem_rate_df, annot=True)
# ax.set_title('3->0 Perturbation Ratio of Successful Attacks')
plt.tight_layout()
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=45, ha='right')
plt.show()
