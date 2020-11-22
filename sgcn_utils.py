import datetime

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from nltk.corpus import stopwords
import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import re
import collections
from collections import Counter
import numpy as np
from multiprocessing import Process, Queue
import pandas as pd
import os
import random
from utils import DataUtil,LogUtil
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.utils import *
from gensim.summarization.textcleaner import *
import tensorflow as tf
import math
from tensorflow.contrib import layers as contrib_layers


def sgcn_mlp_estimator(n_node, hidden_size, comb_dim, n_categories, k, layer_number):
    with tf.variable_scope('data_value_estimator', reuse=tf.AUTO_REUSE):
        # dropout = tf.placeholder(dtype=tf.float32, name="dropout_keep_prob")
        stdv1 = 1.0 / math.sqrt(hidden_size)
        stdv = 1.0 / math.sqrt(n_categories)
        embedding = tf.get_variable('embedding', shape=[n_node + 1, hidden_size], dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(-stdv1, stdv1))
        A = tf.placeholder(dtype=tf.float32, shape=[None, None, None], name="A")
        items = tf.placeholder(dtype=tf.int32, shape=[None, None], name="items")
        alias_input = tf.placeholder(dtype=tf.int32, shape=[None, None], name="alias_input")
        node_masks = tf.placeholder(dtype=tf.int32, shape=[None, None], name="node_masks")
        y_input = tf.placeholder(dtype=tf.float32, shape=[None, n_categories], name="labels")
        y_hat_input = tf.placeholder(dtype=tf.float32, shape=[None, n_categories])

        hidden = tf.nn.embedding_lookup(embedding, items)
        stdv = 1. / math.sqrt(hidden_size)
        w1 = tf.get_variable("w_hidden", shape=[hidden_size, hidden_size],
                             initializer=tf.random_uniform_initializer(-stdv, stdv))
        b1 = tf.get_variable("b_hidden", shape=[hidden_size],
                             initializer=tf.random_uniform_initializer(-stdv, stdv))
        for _ in range(k):
            hidden = tf.matmul(A, hidden)

        def doc_embedding(inputs, node_masks):
            node_masks = tf.expand_dims(node_masks, -1)
            alpha = tf.nn.relu(contrib_layers.fully_connected(inputs, 1))
            zero_vec = -9e15 * tf.ones_like(alpha)
            alpha = tf.where(node_masks > 0, alpha, zero_vec)
            alpha = tf.squeeze(alpha)
            alpha = tf.nn.softmax(alpha)
            alpha = tf.expand_dims(alpha, -1)
            node_masks = tf.cast(node_masks, tf.float32)
            doc_embedding = tf.reduce_sum(tf.multiply(tf.multiply(alpha, inputs), node_masks), 1,
                                          name="doc_embedding")
            return doc_embedding

        seq_hidden = tf.add(tf.matmul(hidden, w1), b1)
        embedding = doc_embedding(seq_hidden, node_masks)

        inputs = tf.concat((embedding, y_input), axis=1)

        w_x_y_concat = tf.get_variable("w_x_y_concat", shape=[hidden_size + 1, hidden_size],
                                       initializer=tf.random_uniform_initializer(-1. / math.sqrt(hidden_size),
                                                                                 1. / math.sqrt(hidden_size)))

        inputs = tf.matmul(inputs, w_x_y_concat)
        inputs = tf.nn.relu(inputs)

        w_x_y_inter = tf.get_variable("w_x_y_inter", shape=[hidden_size, comb_dim],
                                      initializer=tf.random_uniform_initializer(-1. / math.sqrt(comb_dim),
                                                                                1. / math.sqrt(comb_dim)))
        inter_layer = tf.matmul(inputs, w_x_y_inter)
        inter_layer = tf.nn.relu(inter_layer)

        inter_layer = layer_norm(inter_layer)
        comb_layer = tf.concat((inter_layer, y_hat_input), axis=1)
        # with tf.variable_scope("w_x_y_concat"):
        #     inputs = tf.layers.dense(
        #         inputs,
        #         hidden_size,
        #         activation=tf.nn.relu,
        #         kernel_initializer=tf.random_uniform_initializer(-1./ math.sqrt(hidden_size),
        #                                                                1./ math.sqrt(hidden_size)))
        # with tf.variable_scope("w_x_y_inter"):
        #     inter_layer = tf.layers.dense(
        #         inputs,
        #         comb_dim,
        #         activation=tf.nn.relu,
        #         kernel_initializer=tf.random_uniform_initializer(-1./ math.sqrt(comb_dim),
        #                                                                1./ math.sqrt(comb_dim)))

        with tf.variable_scope("w_comb"):
            comb_layer = tf.layers.dense(
                comb_layer,
                comb_dim,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_uniform_initializer(-1. / math.sqrt(comb_dim),
                                                                 1. / math.sqrt(comb_dim)))
        with tf.variable_scope("w_comb_final"):
            dve = tf.layers.dense(
                comb_layer,
                1,
                activation=tf.nn.sigmoid,
                kernel_initializer=tf.random_uniform_initializer(-1. / math.sqrt(1),
                                                                 1. / math.sqrt(1)))
        return dve


class SGCN():
    def __init__(self, hiddenSize, n_node, n_categories, k, gpu):
        
        with tf.device('/cpu:0'):
        #with tf.device('/device:GPU:%d' % gpu):
            self.hidden_size = hiddenSize
            self.n_node = n_node
            self.n_categories = n_categories
            self.dropout = tf.placeholder(dtype=tf.float32, name="dropout_keep_prob")
            self.stdv1 = 1.0 / math.sqrt(self.hidden_size)
            self.stdv = 1.0 / math.sqrt(self.n_categories)
            self.embedding = tf.get_variable('embedding', shape=[n_node + 1, self.hidden_size], dtype=tf.float32,
                                             initializer=tf.random_uniform_initializer(-self.stdv1, self.stdv1))

            self.context = tf.get_variable('context', shape=[1, self.n_categories], dtype=tf.float32,
                                           initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
            self.A = tf.placeholder(dtype=tf.float32, shape=[None, None, None], name="A")
            self.items = tf.placeholder(dtype=tf.int32, shape=[None, None], name="items")
            self.alias_input = tf.placeholder(dtype=tf.int32, shape=[None, None], name="alias_input")
            self.node_masks = tf.placeholder(dtype=tf.int32, shape=[None, None], name="node_masks")
            self.labels = tf.placeholder(dtype=tf.float32, shape=[None, self.n_categories], name="labels")
            self.dropout = tf.placeholder(dtype=tf.float32, name="dropout_keep_prob")
            hidden = tf.nn.embedding_lookup(self.embedding, self.items)

            stdv = 1. / math.sqrt(self.hidden_size)
            w1 = tf.get_variable("w_hidden", shape=[self.hidden_size, self.hidden_size],
                                 initializer=tf.random_uniform_initializer(-stdv, stdv))
            b1 = tf.get_variable("b_hidden", shape=[self.hidden_size],
                                 initializer=tf.random_uniform_initializer(-stdv, stdv))

            stdv = 1. / math.sqrt(self.n_categories)
            w2 = tf.get_variable("w_out", shape=[self.hidden_size, self.n_categories],
                                 initializer=tf.random_uniform_initializer(-stdv, stdv))
            b2 = tf.get_variable("b_out", shape=[self.n_categories],
                                 initializer=tf.random_uniform_initializer(-stdv, stdv))
            for _ in range(k):
                hidden = tf.matmul(self.A, hidden)

            def compute_scores(inputs, node_masks):
                node_masks = tf.expand_dims(node_masks, -1)
                alpha = tf.nn.relu(contrib_layers.fully_connected(inputs, 1))
                zero_vec = -9e15 * tf.ones_like(alpha)
                alpha = tf.where(node_masks > 0, alpha, zero_vec)
                alpha = tf.squeeze(alpha)
                alpha = tf.nn.softmax(alpha)
                alpha = tf.expand_dims(alpha, -1)
                node_masks = tf.cast(node_masks, tf.float32)
                self.doc_embedding = tf.reduce_sum(tf.multiply(tf.multiply(alpha, inputs), node_masks), 1,
                                                   name="doc_embedding")
                tf.logging.info(self.doc_embedding.shape)

                pred = tf.add(tf.matmul(self.doc_embedding, w2), b2)
                return pred

            # hidden = tf.nn.dropout(hidden, self.dropout)
            seq_hidden = tf.add(tf.matmul(hidden, w1), b1)
            logits = tf.nn.softmax(compute_scores(seq_hidden, self.node_masks), name="prediction")
            predictions = tf.argmax(logits, 1)
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=logits)
                self.loss = tf.reduce_mean(losses)
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(predictions, tf.argmax(self.labels, 1))
                self.acc = tf.reduce_mean(tf.cast(correct_predictions, "float"))


def train_sgcn(hidden_size, label_size, n_nodes, sgcn_layer, id_list, all_label, batch_size, model_path,
               step_save_model=50, lr=0.001, epoch=10, window=4, gpu_id=0):
    path = ''
    import tensorflow as tf
    with tf.device('/cpu:0'):
    #with tf.device('/device:GPU:%d' % gpu_id):
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=False,
                log_device_placement=False)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                model = SGCN(hidden_size, n_nodes, label_size, sgcn_layer, 0)
                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(lr)
                grads_and_vars = optimizer.compute_gradients(model.loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
                best_eval_accuracy = 0.0
                saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)
                init = tf.global_variables_initializer()
                sess.run(init)
                for epoch in range(epoch):
                    LogUtil.log("INFO", "epoch:{}".format(epoch))
                    LogUtil.log("INFO", "start training: {}".format(datetime.datetime.now()))
                    x_train_class = SGCNData(id_list, all_label, window)
                    slices = x_train_class.generate_batch(batch_size)
                    for step in range(len(slices)):
                        LogUtil.log('INFO','Training at step:{} '.format(step))
                        i = slices[step]
                        alias_inputs, A, items, node_masks, targets = x_train_class.get_slice(i)
                        targets_onehot = label_to_onehot(targets, label_size)
                        feed_dict = {
                            model.items: items,
                            model.A: A,
                            model.alias_input: alias_inputs,
                            model.node_masks: node_masks,
                            model.labels: targets_onehot,
                            model.dropout: 0.5
                        }
                        _, step, loss, accuracy = sess.run([train_op, global_step, model.loss, model.acc],
                                                           feed_dict)
                        current_step = tf.train.global_step(sess, global_step)
                        if current_step % step_save_model == 0:
                            time_str = datetime.datetime.now().isoformat()
                            LogUtil.log('INFO',
                                "{}: Training step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                            if accuracy > best_eval_accuracy:
                                best_eval_accuracy = accuracy
                                path = saver.save(sess, model_path, global_step=current_step)
                                LogUtil.log('INFO',"Saved model checkpoint to {}\n".format(path))
    return path


def eval_sgcn_doc_embedding(id_list, window, model_path, gpu_id, predict_batch_size=2000):
    import tensorflow as tf
    checkpoint_file = tf.train.latest_checkpoint(model_path)
    LogUtil.log('INFO',"Prepare to load checkpoint from {}".format(checkpoint_file))
    with tf.device('/cpu:0'):
    #with tf.device('/device:GPU:%d' % gpu_id):
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=False,
                log_device_placement=False)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)

                # Get the placeholders from the graph by name
                A = graph.get_operation_by_name("A").outputs[0]
                items = graph.get_operation_by_name("items").outputs[0]
                alias_input = graph.get_operation_by_name("alias_input").outputs[0]
                node_masks = graph.get_operation_by_name("node_masks").outputs[0]
                # input_y = graph.get_operation_by_name("input_y").outputs[0]
                dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

                # Tensors we want to evaluate
                doc_embedding = graph.get_operation_by_name("doc_embedding").outputs[0]
                x_train_class = SGCNData(id_list, window)
                slices = x_train_class.generate_batch(predict_batch_size)
                all_doc_embedding = None
                for step in range(len(slices)):
                    i = slices[step]
                    test_alias_inputs, test_A, test_items, test_tfs, test_node_masks = x_train_class.get_slice(i)
                    feed_dict = {
                        items: test_items,
                        A: test_A,
                        alias_input: test_alias_inputs,
                        node_masks: test_node_masks,
                        dropout_keep_prob: 1.0
                    }
                    doc_embedding_batch = sess.run(doc_embedding, feed_dict)
                    if step == 0:
                        all_doc_embedding = doc_embedding_batch
                    else:
                        all_doc_embedding = np.concatenate((all_doc_embedding, doc_embedding_batch))
                return all_doc_embedding


def eval_sgcn_prediction(id_list, window, model_path, gpu_id, y_test, predict_batch_size=2000):
    LogUtil.log('INFO',"Start inference for sgcn.")
    import tensorflow as tf
    model_dir = os.path.dirname(model_path)
    checkpoint_file = tf.train.latest_checkpoint(model_dir)
    LogUtil.log('INFO',"Prepare to load checkpoint from {}".format(checkpoint_file))
    with tf.device('/cpu:0'):
    #with tf.device('/device:GPU:%d' % gpu_id):
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)
                # Get the placeholders from the graph by name
                A = graph.get_operation_by_name("A").outputs[0]
                items = graph.get_operation_by_name("items").outputs[0]
                alias_input = graph.get_operation_by_name("alias_input").outputs[0]
                node_masks = graph.get_operation_by_name("node_masks").outputs[0]
                # input_y = graph.get_operation_by_name("input_y").outputs[0]
                dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

                # Tensors we want to evaluate
                prediction = graph.get_operation_by_name("prediction").outputs[0]
                x_train_class = SGCNData(id_list, y_test, window)
                slices = x_train_class.generate_batch(predict_batch_size)
                all_predictions = None
                for step in range(len(slices)):
                    i = slices[step]
                    test_alias_inputs, test_A, test_items, test_node_masks, targets = x_train_class.get_slice(i)
                    feed_dict = {
                        items: test_items,
                        A: test_A,
                        alias_input: test_alias_inputs,
                        node_masks: test_node_masks,
                        dropout_keep_prob: 1.0
                    }
                    prediction_batch = sess.run(prediction, feed_dict)
                    if step == 0:
                        all_predictions = prediction_batch
                    else:
                        all_predictions = np.concatenate((all_predictions, prediction_batch))
                # all_predictions = np.argmax(all_predictions, 1)
                if y_test is not None:
                    all_predictions_label = np.argmax(all_predictions, 1)
                    correct_predictions = float(sum(all_predictions_label == y_test))
                    LogUtil.log('INFO',"Total number of test examples: {}".format(len(y_test)))
                    LogUtil.log('INFO',"Accuracy: {:g}".format(correct_predictions / float(len(y_test))))
                return all_predictions


def cle_for_email(sent):
    # Remove Emails
    sent = re.sub('\S*@\S*\s?', '', sent)

    # Remove new line characters
    sent = re.sub('\s+', ' ', sent)

    # Remove distracting single quotes
    sent = re.sub("\'", "", sent)

    return sent


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


def clean_str_simple_version(string, dataset):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    if dataset == '20ng':
        # Remove Emails
        # string = re.sub('\S*@\S*\s?', '', string)

        # Remove new line characters
        string = re.sub('\s+', ' ', string)

    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


def show_statisctic(clean_docs):
    min_len = 10000
    aver_len = 0
    max_len = 0
    num_sentence = sum([len(i) for i in clean_docs])
    ave_num_sentence = num_sentence * 1.0 / len(clean_docs)

    for doc in clean_docs:
        for sentence in doc:
            temp = sentence
            aver_len = aver_len + len(temp)

            if len(temp) < min_len:
                min_len = len(temp)
            if len(temp) > max_len:
                max_len = len(temp)

    aver_len = 1.0 * aver_len / num_sentence

    LogUtil.log('INFO','min_len_of_sentence : ' + str(min_len))
    LogUtil.log('INFO','max_len_of_sentence : ' + str(max_len))
    LogUtil.log('INFO','min_num_of_sentence : ' + str(min([len(i) for i in clean_docs])))
    LogUtil.log('INFO','max_num_of_sentence : ' + str(max([len(i) for i in clean_docs])))
    LogUtil.log('INFO','average_len_of_sentence: ' + str(aver_len))
    LogUtil.log('INFO','average_num_of_sentence: ' + str(ave_num_sentence))
    LogUtil.log('INFO','Total_num_of_sentence : ' + str(num_sentence))

    return max([len(i) for i in clean_docs])


def clean_document(doc_sentence_list, dataset):
    # nltk.download('stopwords')
    stop_words = stopwords.words('english')
    if dataset == '20ng':
        stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
    stop_words = set(stop_words)
    # LogUtil.log('INFO',stop_words)
    stemmer = WordNetLemmatizer()

    word_freq = Counter()

    for doc_sentences in doc_sentence_list:
        for sentence in doc_sentences:
            temp = word_tokenize(clean_str(sentence))  # simple_preprocess(sentence)#
            temp = ' '.join([stemmer.lemmatize(word) for word in temp])

            words = temp.split()
            for word in words:
                word_freq[word] += 1

    highbar = word_freq.most_common(10)[-1][1]
    clean_docs = []
    for doc_sentences in doc_sentence_list:
        clean_doc = []
        count_num = 0
        for sentence in doc_sentences:
            temp = word_tokenize(clean_str(sentence))  # simple_preprocess(sentence)#
            temp = ' '.join([stemmer.lemmatize(word) for word in temp])

            words = temp.split()
            doc_words = []
            for word in words:
                # word not in stop_words and word_freq[word] >= 5
                if dataset == 'mr':
                    if not word in stop_words:
                        doc_words.append(word)
                elif (word not in stop_words) and (word_freq[word] >= 5) and (word_freq[word] < highbar):
                    doc_words.append(word)

            clean_doc.append(doc_words)
            count_num += len(doc_words)

            if dataset == '20ng' and count_num > 2000:
                break

        clean_docs.append(clean_doc)

    return clean_docs


def split_validation(train_set, valid_portion, is_shuffle=False):
    train_set_x = [i for i, j in train_set]
    train_set_y = [j for i, j in train_set]

    if valid_portion == 0.0:
        return (train_set_x, train_set_y)

    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    if is_shuffle:
        np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def count_tf(data):
    rows = []
    cols = []
    vals = []
    for i in range(len(data)):
        for sen in data[i]:
            for j in sen:
                rows.append(i)
                cols.append(j)
                vals.append(1.0)

    # return normalize_features(sp.coo_matrix((vals, (rows, cols)), shape=(len(data), max(cols) + 1))).todense()
    return sp.coo_matrix((vals, (rows, cols)), shape=(len(data), max(cols) + 1)).todense()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def read_file(dataset, topn):
    doc_content_list = []
    doc_sentence_list = []
    f = open('../data/corpus/' + dataset + '.txt', 'rb')
    for line in f.readlines():
        doc_content_list.append(line.strip().decode('latin1'))
        doc_sentence_list.append([i for i in get_sentences(clean_str_simple_version(doc_content_list[-1], dataset))])
    f.close()

    # Remove the rare words
    doc_content_list = clean_document(doc_sentence_list, dataset)
    # LogUtil.log('INFO',doc_content_list)

    # Display the statistics
    max_num_sentence = show_statisctic(doc_content_list)

    word_embeddings_dim = 200
    word_vector_map = {}

    # shulffing
    doc_train_list_original = []
    doc_test_list_original = []
    labels_dic = {}
    label_count = Counter()

    i = 0
    f = open('../data/' + dataset + '.txt', 'r')
    lines = f.readlines()
    for line in lines:
        temp = line.strip().split("\t")
        if temp[1].find('test') != -1:
            doc_test_list_original.append((doc_content_list[i], temp[2]))
        elif temp[1].find('train') != -1:
            doc_train_list_original.append((doc_content_list[i], temp[2]))
        if not temp[2] in labels_dic:
            labels_dic[temp[2]] = len(labels_dic)
        label_count[temp[2]] += 1
        i += 1

    f.close()
    LogUtil.log('INFO',label_count)

    word_freq = Counter()
    word_set = set()
    for doc_words in doc_content_list:
        for words in doc_words:
            for word in words:
                word_set.add(word)
                word_freq[word] += 1

    vocab = list(word_set)
    vocab_size = len(vocab)

    vocab_dic = {}
    for i in word_set:
        vocab_dic[i] = len(vocab_dic) + 1

    LogUtil.log('INFO','Total_number_of_words: ' + str(len(vocab)))
    LogUtil.log('INFO','Total_number_of_categories: ' + str(len(labels_dic)))

    doc_train_list = []
    doc_test_list = []

    for doc, label in doc_train_list_original:
        temp_doc = []
        for sentence in doc:
            temp = []
            for word in sentence:
                temp.append(vocab_dic[word])
            temp_doc.append(temp)
        doc_train_list.append((temp_doc, labels_dic[label]))

    for doc, label in doc_test_list_original:
        temp_doc = []
        for sentence in doc:
            temp = []
            for word in sentence:
                temp.append(vocab_dic[word])
            temp_doc.append(temp)
        doc_test_list.append((temp_doc, labels_dic[label]))

    return doc_content_list, doc_train_list, doc_test_list, vocab_dic, labels_dic, max_num_sentence


def loadGloveModel(gloveFile, vocab_dic, matrix_len):
    LogUtil.log('INFO',"Loading Glove Model")
    f = open(gloveFile, 'r')
    gloveModel = {}
    glove_embedding_dimension = 0
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        glove_embedding_dimension = len(splitLine[1:])
        embedding = np.array([float(val) for val in splitLine[1:]])
        gloveModel[word] = embedding

    words_found = 0
    weights_matrix = np.zeros((matrix_len, glove_embedding_dimension))
    weights_matrix[0] = np.zeros((glove_embedding_dimension,))

    for word in vocab_dic:
        if word in gloveModel:
            weights_matrix[vocab_dic[word]] = gloveModel[word]
            words_found += 1
        else:
            weights_matrix[vocab_dic[word]] = gloveModel[
                'the']  # np.random.normal(scale=0.6, size=(glove_embedding_dimension, ))

    LogUtil.log('INFO',"Total ", len(vocab_dic), " words")
    LogUtil.log('INFO',"Done.", words_found, " words loaded from", gloveFile)

    return weights_matrix


class SGCNData():
    def __init__(self, inputs, targets, window):
        # self.tf_record = count_tf(inputs)
        self.inputs = np.asarray(inputs)
        self.length = len(inputs)
        self.window = window
        self.targets = targets

    def generate_batch(self, batch_size):
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        LogUtil.log('INFO','{} {}'.format(n_batch, batch_size))
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def get_slice(self, iList):
        # inputs = self.inputs[iList]
        inputs, targets = self.inputs[iList], self.targets[iList]
        items, n_node, A, alias_inputs, tfs, node_masks = [], [], [], [], [], []
        mask, node_dic = [], []
        # LogUtil.log('INFO','Flag_5')
        for u_input in inputs:
            temp_s = []
            for s in u_input:
                temp_s += s
            temp_l = list(set(temp_s))
            temp_dic = {temp_l[i]: i for i in range(len(temp_l))}
            n_node.append(temp_l)
            alias_inputs.append([temp_dic[i] for i in temp_s])
            node_dic.append(temp_dic)
        max_n_node = np.max([len(i) for i in n_node])
        # LogUtil.log('INFO','Flag_6')
        for idx in range(len(inputs)):
            node = n_node[idx]
            # tfs.append([self.tf_record[iList[idx], j] for j in list(n_node[idx])] + (max_n_node - len(node)) * [0])
            items.append(node + (max_n_node - len(node)) * [0])
            alias_input = alias_inputs[idx]
            adj = np.zeros((max_n_node, max_n_node))
            for i in range(len(alias_input)):
                for j in alias_input[min(0, i - self.window): max(len(alias_input), i + self.window + 1)]:
                    if not alias_input[i] == j:
                        adj[alias_input[i]][j] += 1.0
            adj = sp.csr_matrix(adj)
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
            adj = normalize_adj(adj + sp.eye(adj.shape[0]))
            A.append(np.asarray(adj.todense()))
            alias_inputs[idx] = [j for j in range(max_n_node)]
            node_masks.append([1 for j in node] + (max_n_node - len(node)) * [0])
        return alias_inputs, A, items, node_masks, targets


class Data():
    def __init__(self, data, window, num_categories):
        inputs = data[0]
        self.tf_record = count_tf(inputs)
        self.inputs = np.asarray(inputs)
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.window = window
        self.num_categories = num_categories

    def generate_batch(self, batch_size, shuffle=False):
        if shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
            self.tf_record = self.tf_record[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def get_slice(self, iList):
        inputs, targets = self.inputs[iList], self.targets[iList]
        items, n_node, A, alias_inputs, tfs, node_masks = [], [], [], [], [], []
        mask, node_dic = [], []
        for u_input in inputs:
            temp_s = []
            for s in u_input:
                temp_s += s
            temp_l = list(set(temp_s))
            temp_dic = {temp_l[i]: i for i in range(len(temp_l))}
            n_node.append(temp_l)
            alias_inputs.append([temp_dic[i] for i in temp_s])
            node_dic.append(temp_dic)
        max_n_node = np.max([len(i) for i in n_node])
        for idx in range(len(inputs)):
            node = n_node[idx]
            tfs.append([self.tf_record[iList[idx], j] for j in list(n_node[idx])] + (max_n_node - len(node)) * [0])
            items.append(node + (max_n_node - len(node)) * [0])
            alias_input = alias_inputs[idx]
            adj = np.zeros((max_n_node, max_n_node))
            for i in range(len(alias_input)):
                for j in alias_input[min(0, i - self.window): max(len(alias_input), i + self.window + 1)]:
                    if not alias_input[i] == j:
                        adj[alias_input[i]][j] += 1.0
            adj = sp.csr_matrix(adj)
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
            adj = normalize_adj(adj + sp.eye(adj.shape[0]))
            A.append(np.asarray(adj.todense()))
            alias_inputs[idx] = [j for j in range(max_n_node)]
            node_masks.append([1 for j in node] + (max_n_node - len(node)) * [0])
        return alias_inputs, A, items, targets, tfs, node_masks


def generate_id_for_sgcn(pd_train_list, vocab_dic, dataset):
    doc_content_list = []
    doc_sentence_list = []
    for line in pd_train_list:
        doc_content_list.append(line.strip())
        doc_sentence_list.append([i for i in get_sentences(clean_str_simple_version(doc_content_list[-1], dataset))])
    doc_content_list = clean_document(doc_sentence_list, dataset)
    # max_num_sentence = show_statisctic(doc_content_list)
    doc_id_list = []
    for doc in doc_content_list:
        temp_doc = []
        for sentence in doc:
            temp = []
            for word in sentence:
                if word in vocab_dic:
                    temp.append(vocab_dic[word])
                else:
                    temp.append(0)
            temp_doc.append(temp)
        doc_id_list.append(temp_doc)
    return doc_id_list


def label_to_onehot(targets, n_label):
    # targets = [int(x) - 1 for x in list(targets)]
    targets = np.asarray(targets)
    targets = np.eye(n_label)[targets.astype(int)]
    return targets


def layer_norm(input_tensor, name=None):
    """Run layer normalization on the last dimension of the tensor."""
    if input_tensor.dtype == tf.float16:
        try:
            from modeling.fused_layer_norm import fused_layer_norm
            return fused_layer_norm(
                inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name,
                use_fused_batch_norm=True)
        except ImportError:
            return tf.contrib.layers.layer_norm(
                inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)
    else:
        return tf.contrib.layers.layer_norm(
            inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def layer_norm_and_dropout(input_tensor, dropout_prob, name=None):
    """Runs layer normalization followed by dropout."""
    output_tensor = layer_norm(input_tensor, name)
    output_tensor = dropout(output_tensor, dropout_prob)
    return output_tensor


def dropout(input_tensor, dropout_prob):
    """Perform dropout.

    Args:
      input_tensor: float Tensor.
      dropout_prob: Python float. The probability of dropping out a value (NOT of
        *keeping* a dimension as in `tf.nn.dropout`).

    Returns:
      A version of `input_tensor` with dropout applied.
    """
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor

    output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
    return output
