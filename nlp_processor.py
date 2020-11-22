import io
import numpy as np
import gensim
import configparser
import random
import re
import datetime
import nltk
import pandas as pd
from nltk.stem import SnowballStemmer
from utils import DataUtil, LogUtil
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer, TfidfVectorizer
from collections import defaultdict
from string import punctuation

stops = set(stopwords.words("english"))
stop_words = ['the','a','an','and','but','if','or','because','as','what','which','this','that','these','those','then',
              'just','so','than','such','both','through','about','for','is','of','while','during','to','What','Which',
              'Is','If','While','This']


class TfidfEmbeddingVectorizer:
    def __init__(self, word2vec, dim):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = dim
        print('Self dim', self.dim)
        self.digit = re.compile(r'(\d+)')

    def preproc(self, text):
        return [
            re.sub('\W+', '', t) for t in text.split() if not (t.isspace() or self.digit.search(t) or t in punctuation)
        ]

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x, tokenizer=self.preproc, stop_words='english', max_df=.95,
                                min_df=2, binary=True)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        X_clean =[TextPreProcessor.clean_str(x) for x in X]
        return np.array([
            np.mean([self.word2vec[w] * self.word2weight[w]
                     for w in words.split() if w not in punctuation and w in self.word2vec] or
                    [np.zeros(self.dim)], axis=0)
            for words in X_clean
        ])

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)



class BagOfWords(object):
    def __init__(self):
        pass

    @staticmethod    
    def generateBOW(df_features,vocab_size):
        now = datetime.datetime.now()
        print(now.strftime('%Y-%m-%d %H:%M:%S'))        
        LogUtil.log("INFO", "Start to generate attribute BOW!")
        BagOfWordsExtractor = CountVectorizer(max_features=vocab_size,
                                            analyzer='word',
                                            lowercase=True)
        bow_features = BagOfWordsExtractor.fit_transform(df_features)
        print(now.strftime('%Y-%m-%d %H:%M:%S'))
        LogUtil.log("INFO", "End to generate attribute BOW!")        
        return bow_features.toarray()

    @staticmethod    
    def generateBOW_with_vocab(df_features,vocabPath):
        now = datetime.datetime.now()
        print(now.strftime('%Y-%m-%d %H:%M:%S'))        
        LogUtil.log("INFO", "Start to generate attribute BOW!")
        vocab = list()
        with open(vocabPath,'r',encoding='utf8') as fi:
            for line in fi:
                segs = line.rstrip().split('\t')
                if len(segs) <1:
                    continue
                vocab.append(segs[0])

        BagOfWordsExtractor = CountVectorizer(vocabulary=vocab,
                                            analyzer='word',
                                            lowercase=True)
        bow_features = BagOfWordsExtractor.fit_transform(df_features)
        print(now.strftime('%Y-%m-%d %H:%M:%S'))
        LogUtil.log("INFO", "End to generate attribute BOW!")        
        return bow_features.toarray()
    
    @staticmethod
    def generateBOW_charLevel(df_features):
        now = datetime.datetime.now()
        print(now.strftime('%Y-%m-%d %H:%M:%S'))
        LogUtil.log("INFO", "Start to generate attribute BOW at Char level!")        
        maxNumFeatures = 4001
        # bag of letter sequences (chars)
        BagOfWordsExtractor = CountVectorizer(max_df=1.0, min_df=1, max_features=maxNumFeatures,
                                            analyzer='char', ngram_range=(1, 3),
                                            binary=True, lowercase=True)
        trainQuestion1_BOW_rep = BagOfWordsExtractor.transform(df_features.ix[:, 'question1'])
        trainQuestion2_BOW_rep = BagOfWordsExtractor.transform(df_features.ix[:, 'question2'])
        X = -(trainQuestion1_BOW_rep != trainQuestion2_BOW_rep).astype(int)
        df_features['f_bag_words'] = [X[i, :].toarray()[0] for i in range(0, len(df_features))]
        for j in range(0, len(df_features['f_bag_words'][0])):
            df_features['z_bag_words' + str(j)] = [df_features['f_bag_words'][i][j] for i in range(0, len(df_features))]
        df_features.fillna(0.0)
        now = datetime.datetime.now()
        print(now.strftime('%Y-%m-%d %H:%M:%S'))
        LogUtil.log("INFO", "Finish to generate attribute BOW at Char level!")        
        return df_features
    
class TextPreProcessor(object):

    _stemmer = SnowballStemmer('english')

    def __init__(self):
        pass

    @staticmethod
    def remove_stopwords(text, method):
        text = text.split()
        if method == 'nltk':
            text = [w for w in text if not w in stops]
        else:
            text = [w for w in text if not w in stop_words]
        text = ' '.join(text)
        return  text

    @staticmethod
    def clean_str(string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    @staticmethod
    def clean_text(text):
        """
        Clean text
        :param text: the string of text
        :return: text string after cleaning
        """
        # unit
        text = re.sub(r"(\d+)kgs ", lambda m: m.group(1) + ' kg ', text)        # e.g. 4kgs => 4 kg
        text = re.sub(r"(\d+)kg ", lambda m: m.group(1) + ' kg ', text)         # e.g. 4kg => 4 kg
        text = re.sub(r"(\d+)k ", lambda m: m.group(1) + '000 ', text)          # e.g. 4k => 4000
        text = re.sub(r"\$(\d+)", lambda m: m.group(1) + ' dollar ', text)
        text = re.sub(r"(\d+)\$", lambda m: m.group(1) + ' dollar ', text)

        # acronym
        text = re.sub(r"can\'t", "can not", text)
        text = re.sub(r"cannot", "can not ", text)
        text = re.sub(r"what\'s", "what is", text)
        text = re.sub(r"What\'s", "what is", text)
        text = re.sub(r"\'ve ", " have ", text)
        text = re.sub(r"n\'t", " not ", text)
        text = re.sub(r"i\'m", "i am ", text)
        text = re.sub(r"I\'m", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r"c\+\+", "cplusplus", text)
        text = re.sub(r"c \+\+", "cplusplus", text)
        text = re.sub(r"c \+ \+", "cplusplus", text)
        text = re.sub(r"c#", "csharp", text)
        text = re.sub(r"f#", "fsharp", text)
        text = re.sub(r"g#", "gsharp", text)
        text = re.sub(r" e mail ", " email ", text)
        text = re.sub(r" e \- mail ", " email ", text)
        text = re.sub(r" e\-mail ", " email ", text)
        text = re.sub(r",000", '000', text)
        text = re.sub(r"\'s", " ", text)

        # spelling correction
        text = re.sub(r"ph\.d", "phd", text)
        text = re.sub(r"PhD", "phd", text)
        text = re.sub(r"pokemons", "pokemon", text)
        text = re.sub(r"pokémon", "pokemon", text)
        text = re.sub(r"pokemon go ", "pokemon-go ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" 9 11 ", " 911 ", text)
        text = re.sub(r" j k ", " jk ", text)
        text = re.sub(r" fb ", " facebook ", text)
        text = re.sub(r"facebooks", " facebook ", text)
        text = re.sub(r"facebooking", " facebook ", text)
        text = re.sub(r"insidefacebook", "inside facebook", text)
        text = re.sub(r"donald trump", "trump", text)
        text = re.sub(r"the big bang", "big-bang", text)
        text = re.sub(r"the european union", "eu", text)
        text = re.sub(r" usa ", " america ", text)
        text = re.sub(r" us ", " america ", text)
        text = re.sub(r" u s ", " america ", text)
        text = re.sub(r" U\.S\. ", " america ", text)
        text = re.sub(r" US ", " america ", text)
        text = re.sub(r" American ", " america ", text)
        text = re.sub(r" America ", " america ", text)
        text = re.sub(r" quaro ", " quora ", text)
        text = re.sub(r" mbp ", " macbook-pro ", text)
        text = re.sub(r" mac ", " macbook ", text)
        text = re.sub(r"macbook pro", "macbook-pro", text)
        text = re.sub(r"macbook-pros", "macbook-pro", text)
        text = re.sub(r" 1 ", " one ", text)
        text = re.sub(r" 2 ", " two ", text)
        text = re.sub(r" 3 ", " three ", text)
        text = re.sub(r" 4 ", " four ", text)
        text = re.sub(r" 5 ", " five ", text)
        text = re.sub(r" 6 ", " six ", text)
        text = re.sub(r" 7 ", " seven ", text)
        text = re.sub(r" 8 ", " eight ", text)
        text = re.sub(r" 9 ", " nine ", text)
        text = re.sub(r"googling", " google ", text)
        text = re.sub(r"googled", " google ", text)
        text = re.sub(r"googleable", " google ", text)
        text = re.sub(r"googles", " google ", text)
        text = re.sub(r" rs(\d+)", lambda m: ' rs ' + m.group(1), text)
        text = re.sub(r"(\d+)rs", lambda m: ' rs ' + m.group(1), text)
        text = re.sub(r"the european union", " eu ", text)
        text = re.sub(r"dollars", " dollar ", text)

        # punctuation
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"-", " - ", text)
        text = re.sub(r"/", " / ", text)
        text = re.sub(r"\\", " \ ", text)
        text = re.sub(r"=", " = ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r"\.", " . ", text)
        text = re.sub(r",", " , ", text)
        text = re.sub(r"\?", " ? ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\"", " \" ", text)
        text = re.sub(r"&", " & ", text)
        text = re.sub(r"\|", " | ", text)
        text = re.sub(r";", " ; ", text)
        text = re.sub(r"\(", " ( ", text)
        text = re.sub(r"\)", " ( ", text)

        # symbol replacement
        text = re.sub(r"&", " and ", text)
        text = re.sub(r"\|", " or ", text)
        text = re.sub(r"=", " equal ", text)
        text = re.sub(r"\+", " plus ", text)
        text = re.sub(r"₹", " rs ", text)      # 测试！
        text = re.sub(r"\$", " dollar ", text)

        # remove extra space
        text = ' '.join(text.split())
        return text

    @staticmethod
    def stem(df):
        """
        Process the text data with SnowballStemmer
        :param df: dataframe of original data
        :return: dataframe after stemming
        """
        df['englishQ_1'] = df.englishQ_1.map(lambda x: ' '.join(
            [TextPreProcessor._stemmer.stem(word) for word in
             nltk.word_tokenize(TextPreProcessor.clean_text(str(x).lower()))]))
        df['englishQ_2'] = df.englishQ_2.map(lambda x: ' '.join(
            [TextPreProcessor._stemmer.stem(word) for word in
             nltk.word_tokenize(TextPreProcessor.clean_text(str(x).lower()))]))
        return df


class Word2Vec(object):
    def __init__(self,word2vec,dim):
        self.dim= dim
        self.word2vec = word2vec
        print('Self dim', self.dim)

    @staticmethod
    def generateW2V(df_features):
        model = gensim.models.KeyedVectors.load('../data/GoogleNews_vectors-negative300_uncompress_noL2Norm',mmap='r')
        all_vectors = list()
        for feature in df_features:
            sum_vector = np.zeros(300)
            all_words = feature.split()
            valid_words = 0
            for word in all_words:
                if word in model.vocab:
                    valid_words += 1
                    cur_vector = model[word]
                    sum_vector = np.sum([sum_vector,cur_vector],axis=0)
            if valid_words == 0:
                print(feature)
                avg_vector = sum_vector
            else:
                avg_vector = np.divide(sum_vector,valid_words)
            all_vectors.append(avg_vector)
        return  np.asmatrix(all_vectors)

    def transform(self, X):
        X_clean =[TextPreProcessor.clean_str(x) for x in X]
        return np.array([
            np.mean([self.word2vec[w]
                     for w in words.split() if w not in punctuation and w in self.word2vec] or
                    [np.zeros(self.dim)], axis=0)
            for words in X_clean
        ])

class word_embedding_processor(object):
    def load_vectors(fname):
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = list(map(int, fin.readline().split()))
        data = {}
        for index, line in enumerate(fin):
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = np.asarray(list(map(float, tokens[1:])))
        return data


if __name__== '__main__':
    # with open('../data/cikm_english_train_20180516.txt','r',encoding='utf-8') as fi:
    #     for line in fi:
    #         TextPreProcessor.stem()
    # df = pd.read_csv('../data/cikm_english_train_20180516.txt','\t')
    # df.columns = ['englishQ_1','spanishT_1','englishQ_2','spanishT_2','label']
    # TextPreProcessor.stem(df)
    # for k,v in df.items():
    #     print(k)
    print(len(Word2Vec.generateW2V()))
