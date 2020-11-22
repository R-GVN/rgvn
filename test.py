import multiprocessing
import os
import gensim
import tensorflow as tf
import pandas as pd
import numpy as np
from nlp_processor import Word2Vec
from string import punctuation
from utils import PlotUtils
import lightgbm
import shutil
from utils import DataUtil

def lightGBM_test():
    model =lightgbm.LGBMClassifier()
    paras={"objective":"multiclass","num_class":4}
    model.set_params(objective="multiclass")
    model.set_params(num_class=4)
    print(model.get_params())

def plot_test():
    temp_out = np.loadtxt("./metrics/lightGBM_performance_word2vec_0.0",delimiter=',')
    print(temp_out)
    PlotUtils.plot_graph(temp_out,20)
    # lowest_val = np.load("./metrics/noisy_index_word2vec_0.2.npy")
    # np.savetxt("./metrics/noisy_index_word2vec_0.2.txt",lowest_val,fmt='%f', delimiter=',')
    # print(lowest_val)
def test_tf(model_dir):
    path = tf.train.latest_checkpoint(model_dir)
    return path
def dataframe_test():
    # print(test_tf(model_path))
    df = pd.DataFrame([[1,2,3],[2,3,4],[3,4,5]])
    df.columns=['A','B','C']
    df_matrix = np.matrix(df)
    df_ndarrary = df.to_numpy()
    # print(df_ndarrary[0,:])
    # print(df_matrix[0,:])
    df_text = pd.DataFrame([["hellow world hello"],["hellow world"]])
    df_text.columns =['text']
    # np.array([ w+"_result" [for w in words]for words in df_text])
    print(df_text.text.tolist())
    result = [max([w for w in words.split()])for words in df_text.text.tolist()]

def word2vec_test():
    # model = gensim.models.KeyedVectors.load_word2vec_format( \
    #
    # model.init_sims(replace=False)
    word_vecs = gensim.models.KeyedVectors.load('../data/GoogleNews_vectors-negative300_uncompress_noL2Norm',mmap='r')
    df_text = pd.DataFrame([["hellow world hello"],["hellow world"]])
    df_text.columns =['text']
    terms =['hellow','world','hello']
    terms_vectors = [(term,word_vecs[term]) if term in word_vecs else "Null" for term in terms ]
    print(terms_vectors)
    word2vec = Word2Vec(word_vecs,300)
    # result = Word2Vec.generateW2V(df_text.text.tolist())
    result = word2vec.transform(df_text.text.tolist())
    print(result)
def delete_files(out_dir):
    for filename in os.listdir(out_dir):
        file_path = os.path.join(out_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def cal_multiple(x):
    return tf.multiply(x,x)

def cal_sum(x,y,gpu):
    with tf.device('/cpu:0'):
        z = tf.add(x,y)
    with tf.device('/device:GPU:%d' % gpu):
        result = tf.add(x,z)
    return result

if __name__=="__main__":
    sess = tf.InteractiveSession()
    with sess.as_default():
        x=tf.constant([1,1,1])
        y = tf.constant([1, 1, 1])
        result = cal_sum(x,y,0)
        print(result.eval())
