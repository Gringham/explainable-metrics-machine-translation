import json

def load_files(files):
    '''loads json attributions'''
    attributions = []
    for file in files:
        with open(file, 'r') as f:
            attributions += json.loads(f.read())

    return attributions

def print_attributions(attributions):
    '''
    :param attributions: json format attributions to print out
    :return:
    '''
    for attribution in attributions:
        print("Sample: ", attribution['corpus_row'])
        print("SRC:", attribution['src'])
        print("REF:", attribution['ref'])
        print("HYP:", attribution['hyp'])
        for key, value in attribution['metrics'].items():
            print(key, 'Score:', value['score'], 'Attributions:', value['attributions'])
        print('\n')

if __name__ == '__main__':
    files = ['0_0_mlqe_attributions_margin.json']
    print_attributions(load_files(files))