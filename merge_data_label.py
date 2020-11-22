from nltk.corpus import stopwords
import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from utils import clean_str, loadWord2Vec
import sys

if len(sys.argv) != 2:
	sys.exit("Use: python merge_data_label.py <dataset>")

datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr']
dataset = sys.argv[1]

if dataset not in datasets:
	sys.exit("wrong dataset name")

label_map = dict()

f_label_map = open('data/corpus/' + dataset + '_labels.txt', 'r')
# f = open('data/wiki_long_abstracts_en_text.txt', 'r')
label_id = 0
for line in f_label_map.readlines():
    label_map[line.rstrip()] = label_id
    label_id +=1

contents = list()
f_content = open('data/corpus/' + dataset + '.clean.txt', 'r')
# f = open('data/wiki_long_abstracts_en_text.txt', 'r')
for line in f_content.readlines():
    contents.append(line.strip())


f_content_label_map = open('data/' + dataset + '.txt', 'r')
# f = open('data/wiki_long_abstracts_en_text.txt', 'r')
idx = 0
doc_test = dict()
doc_train = dict()
for line in f_content_label_map.readlines():
    segs = line.rstrip().split('\t')
    label = label_map[segs[2]]
    if segs[1].find('test') != -1:
        doc_test[idx] = label
    elif segs[1].find('train') != -1:
        doc_train[idx] = label
    idx += 1

doc_test_final = list()
doc_train_final = list()
for key in doc_test:
    doc_test_final.append(str(doc_test[key])+"\t"+contents[int(key)])

for key in doc_train:
    doc_train_final.append(str(doc_train[key])+"\t"+contents[int(key)])

f = open('data/corpus/' + dataset + '_test.txt', 'w')
f.write("\n".join(doc_test_final))
f.close()

f = open('data/corpus/' + dataset + '_train.txt', 'w')
f.write("\n".join(doc_train_final))
f.close()