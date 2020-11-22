rom __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensorflow.python.tools import inspect_checkpoint as chkp
import numpy as np
import tensorflow as tf
import argparse

def corrupt_label(y_train, noise_rate):
    """Corrupts training labels.

    Args:
      y_train: training labels
      noise_rate: input noise ratio

    Returns:
      corrupted_y_train: corrupted training labels
      noise_idx: corrupted index
    """

    y_set = list(set(y_train))

    # Sets noise_idx
    temp_idx = np.random.permutation(len(y_train))
    noise_idx = temp_idx[:int(len(y_train) * noise_rate)]

    # Corrupts label
    corrupted_y_train = y_train[:]

    for itt in noise_idx:
        temp_y_set = y_set[:]
        del temp_y_set[y_train[itt]]
        rand_idx = np.random.randint(len(y_set) - 1)
        corrupted_y_train[itt] = temp_y_set[rand_idx]

    return corrupted_y_train, noise_idx

def corrupt_label_uniform(y_train, noise_rate):
    """Corrupts training labels.

    Args:
      y_train: training labels
      noise_rate: input noise ratio

    Returns:
      corrupted_y_train: corrupted training labels
      noise_idx: corrupted index
    """

    y_set = list(set(y_train))

    # Sets noise_idx
    temp_idx = np.random.permutation(len(y_train))
    noise_idx = temp_idx[:int(len(y_train) * noise_rate)]

    # Corrupts label
    corrupted_y_train = y_train[:]

    for itt in noise_idx:
        temp_y_set = y_set[:]
        del temp_y_set[y_train[itt]]
        rand_idx = np.random.randint(len(y_set) - 1)
        corrupted_y_train[itt] = temp_y_set[rand_idx]

    return corrupted_y_train, noise_idx

def ckpt_modifier(checkpoint_path,new_checkpoint_path):
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    with tf.Session() as sess:
        new_var_list = []
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_path):
            var = tf.contrib.framework.load_variable(checkpoint_path, var_name)
            # if 'word_embeddings' in var_name:
            # if "layer_3" in var_name or "layer_4" in var_name or "layer_5" in var_name:
            #     new_var_name = var_name.replace("layer_3","layer_0").replace("layer_4","layer_1").replace("layer_5","layer_2")
            #     named_var = tf.Variable(var, name=new_var_name)
            #     new_var_list.append(named_var)
            named_var = None
            if 'loss' in var_name:
                continue
            if "meta/embeddings/word_embeddings" in var_name:
                named_var = tf.Variable(var, name="bert/token_embeddings/word_embeddings")
            else:
                named_var = tf.Variable(var, name=var_name)
            new_var_list.append(named_var)

        LogUtil.log('INFO','starting to write new checkpoint !')
        saver = tf.train.Saver(var_list=new_var_list)
        sess.run(tf.global_variables_initializer())
        model_name = 'share_word_embedding_query_meta_layer12'
        checkpoint_path = os.path.join(new_checkpoint_path, model_name)
        saver.save(sess, checkpoint_path)
        LogUtil.log('INFO',"done !")
def print_all_variable_from_model(model_path):
    chkp.print_tensors_in_checkpoint_file(model_path,tensor_name='',all_tensors=True)

def check_variable_from_model(model_path,variable):
    chkp.print_tensors_in_checkpoint_file(model_path,tensor_name=variable,all_tensors=False)

if __name__=="__main__":
    model_path = "./tmp/model.ckpt"
    # variable = 'bert/idf_embeddings'
    # check_variable_from_model(model_path,variable)
    print_all_variable_from_model(model_path)
