import os, shutil
import requests
import tarfile

def download_wmt():
    urls = {'wmt17': ('http://data.statmt.org/wmt17/translation-task/wmt17-submitted-data-v1.0.tgz', 'wmt17-submitted-data'),
            'wmt18': ('http://data.statmt.org/wmt18/translation-task/wmt18-submitted-data-v1.0.tgz', 'wmt18-submitted-data'),
            'wmt19': ('http://data.statmt.org/wmt19/translation-task/wmt19-submitted-data-v3.tgz', 'wmt19-submitted-data'),
            'wmt20': ('http://data.statmt.org/wmt20/translation-task/wmt20-submitted-systems.tgz', 'wmt20-submitted-systems')}

    for key in urls:
        url = urls[key][0]
        if not os.path.exists(key):
            os.makedirs(key)

        response = requests.get(url, stream=True)
        file = tarfile.open(fileobj=response.raw, mode="r|gz")
        file.extractall(path=key)

        path = os.path.join(key, urls[key][1], 'txt')
        files_list = os.listdir(path)
        for files in files_list:
            shutil.move(os.path.join(path, files), key)

def download_da():
    urls = {'wmt17': (
    'http://data.statmt.org/wmt17/translation-task/wmt17-submitted-data-v1.0.tgz', 'wmt17-submitted-data'),
            'wmt18': (
            'http://data.statmt.org/wmt18/translation-task/wmt18-submitted-data-v1.0.tgz', 'wmt18-submitted-data'),
            'wmt19': (
            'http://ufallab.ms.mff.cuni.cz/~bojar/wmt19-metrics-task-package.tgz', 'wmt19-submitted-data'),
            'wmt20': (
            'http://data.statmt.org/wmt20/translation-task/wmt20-submitted-systems.tgz', 'wmt20-news-task-primary-submissions')}

    for key in urls:
        url = urls[key][0]
        if not os.path.exists(key):
            os.makedirs(key)

        response = requests.get(url, stream=True)
        file = tarfile.open(fileobj=response.raw, mode="r|gz")
        file.extractall(path=key)

        path = os.path.join(key, urls[key][1], 'txt')
        files_list = os.listdir(path)
        for files in files_list:
            shutil.move(os.path.join(path, files), key)

if __name__ == '__main__':
    download_da()