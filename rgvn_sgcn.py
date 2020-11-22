
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
import numpy as np
from sklearn import metrics
import tensorflow.compat.v1 as tf
import tqdm
import rgvn_metrics
from tensorflow.contrib import layers as contrib_layers
from sgcn_utils import sgcn_mlp_estimator, SGCNData, train_sgcn, eval_sgcn_prediction, layer_norm,label_to_onehot
import math
from utils import DataUtil,LogUtil

class RgvnSGCN(object):
    """Data Valuation using Reinforcement Learning (DVRL) class.

      Attributes:
        x_train: training feature
        y_train: training labels
        x_valid: validation features
        y_valid: validation labels
        problem: 'regression' or 'classification'
        pred_model: predictive model (object)
        parameters: network parameters such as hidden_dim, iterations,
                    activation function, layer_number, learning rate
        checkpoint_file_name: File name for saving and loading the trained model
        flags: flag for training with stochastic gradient descent (flag_sgd)
               and flag for using pre-trained model (flag_pretrain)
    """

    def __init__(self, n_nodes, x_train, y_train, x_valid, y_valid,
                 problem, pred_model, parameters, checkpoint_file_name, flags):
        """Initializes DVRL."""

        # Inputs
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

        self.problem = problem

        # One-hot encoded labels
        if self.problem == 'classification':
            self.y_train_onehot = \
                np.eye(len(np.unique(y_train)))[y_train.astype(int)]
            self.y_valid_onehot = \
                np.eye(len(np.unique(y_train)))[y_valid.astype(int)]
        elif self.problem == 'regression':
            self.y_train_onehot = np.reshape(y_train, [len(y_train), 1])
            self.y_valid_onehot = np.reshape(y_valid, [len(y_valid), 1])

        # Network parameters
        self.hidden_dim = parameters['hidden_dim']
        self.comb_dim = parameters['comb_dim']
        self.outer_iterations = parameters['iterations']
        self.act_fn = parameters['activation']
        self.layer_number = parameters['layer_number']
        self.n_nodes = n_nodes
        self.batch_size = np.min(
            [parameters['batch_size'], len(x_train)])
        self.learning_rate = parameters['learning_rate']

        # Basic parameters
        self.epsilon = 1e-8  # Adds to the log to avoid overflow
        self.threshold = 0.9  # Encourages exploration

        # Flags
        self.flag_sgd = flags['sgd']
        self.flag_pretrain = flags['pretrain']

        # If the pred_model uses stochastic gradient descent (SGD) for training
        if self.flag_sgd:
            self.inner_iterations = parameters['inner_iterations']
            self.batch_size_predictor = np.min([parameters['batch_size_predictor'],
                                                len(x_valid)])

        # Checkpoint file name
        self.checkpoint_file_name = checkpoint_file_name

        # Basic parameters
        self.data_dim = parameters['hidden_dim']
        self.label_dim = len(self.y_train_onehot[0, :])

        # Training Inputs
        # x_input can be raw input or its encoded representation, e.g. using a
        # pre-trained neural network. Using encoded representation can be beneficial
        # to reduce computational cost for high dimensional inputs, like images.
        self.A = tf.placeholder(dtype=tf.float32, shape=[None, None, None], name="A")
        self.items = tf.placeholder(dtype=tf.int32, shape=[None, None], name="items")
        self.alias_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="alias_input")
        self.node_masks = tf.placeholder(dtype=tf.int32, shape=[None, None], name="node_masks")
        self.x_input = tf.placeholder(tf.float32, [None, self.data_dim])
        self.y_input = tf.placeholder(tf.float32, [None, self.label_dim])

        # Prediction difference
        # y_hat_input is the prediction difference between predictive models
        # trained on the training set and validation set.
        # (adding y_hat_input into data value estimator as the additional input
        # is observed to improve data value estimation quality in some cases)
        self.y_hat_input = tf.placeholder(tf.float32, [None, self.label_dim])

        # Selection vector
        self.s_input = tf.placeholder(tf.float32, [None, 1])

        # Rewards (Reinforcement signal)
        self.reward_input = tf.placeholder(tf.float32)

        # Pred model (Note that any model architecture can be used as the predictor
        # model, either randomly initialized or pre-trained with the training data.
        # The condition for predictor model to have fit (e.g. using certain number
        # of back-propagation iterations) and predict functions as its subfunctions.
        self.pred_model = pred_model

        # Final model
        self.final_model = pred_model

        self.ori_model = self.pred_model
        if not os.path.exists('tmp'):
            os.makedirs('tmp')
        # Baseline model
        LogUtil.log('INFO',"Start to train SGCN DVRL original baseline prediction model")
        self.ori_model_path = train_sgcn(self.hidden_dim, self.label_dim, self.n_nodes, 1, self.x_train, self.y_train,
                                         self.batch_size_predictor, \
                                         'tmp/sgcn_as_predict_baseline_model', step_save_model=4, lr=0.001,
                                         epoch=self.inner_iterations)
        LogUtil.log('INFO',"Save SGCN DVRL baseline prediction model")

        # Valid baseline model
        LogUtil.log('INFO',"Start to train SGCN DVRL valid prediction model")
        self.val_model_path = train_sgcn(self.hidden_dim, self.label_dim, self.n_nodes, 1, self.x_valid, self.y_valid,
                                         int(self.batch_size_predictor/2), \
                                         'tmp/sgcn_as_predict_validation_model', step_save_model=40, lr=0.001,
                                         epoch=self.inner_iterations)
        LogUtil.log('INFO',"Save SGCN DVRL valid baseline prediction model")

        self.final_model_path = None

    def rpm(self):
        # return sgcn_mlp_estimator(self.n_nodes,self.hidden_dim,self.comb_dim,self.label_dim,1,self.layer_number)
        hidden_size = self.hidden_dim
        n_categories = self.label_dim
        n_node = self.n_nodes
        comb_dim = self.comb_dim
        with tf.device('/cpu:0'),tf.variable_scope('data_value_estimator', reuse=tf.AUTO_REUSE):
            # dropout = tf.placeholder(dtype=tf.float32, name="dropout_keep_prob")
            stdv1 = 1.0 / math.sqrt(hidden_size)
            stdv = 1.0 / math.sqrt(n_categories)
            embedding = tf.get_variable('embedding', shape=[n_node + 1, hidden_size], dtype=tf.float32,
                                        initializer=tf.random_uniform_initializer(-stdv1, stdv1))
            hidden = tf.nn.embedding_lookup(embedding, self.items)
            stdv = 1. / math.sqrt(hidden_size)
            w1 = tf.get_variable("w_hidden", shape=[hidden_size, hidden_size],
                                 initializer=tf.random_uniform_initializer(-stdv, stdv))
            b1 = tf.get_variable("b_hidden", shape=[hidden_size],
                                 initializer=tf.random_uniform_initializer(-stdv, stdv))
            for _ in range(self.layer_number):
                hidden = tf.matmul(self.A, hidden)

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
            embedding = doc_embedding(seq_hidden, self.node_masks)
            inputs = tf.concat((embedding, self.y_input), axis=1)
            w_x_y_concat = tf.get_variable("w_x_y_concat", shape=[hidden_size + n_categories, hidden_size],
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
            comb_layer = tf.concat((inter_layer, self.y_hat_input), axis=1)

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
   
    def train_rgvn(self, perf_metric):
        """Trains DVRL based on the specified objective function.

        Args:
          perf_metric: 'auc', 'accuracy', 'log-loss' for classification
                       'mae', 'mse', 'rmspe' for regression
        """

        # Generates selected probability
        est_data_value = self.rpm()

        # Generator loss (REINFORCE algorithm)
        prob = tf.reduce_sum(self.s_input * tf.log(est_data_value + self.epsilon) +
                             (1 - self.s_input) *
                             tf.log(1 - est_data_value + self.epsilon))
        dve_loss = (-self.reward_input * prob) + \
                   1e3 * (tf.maximum(tf.reduce_mean(est_data_value)
                                     - self.threshold, 0) +
                          tf.maximum((1 - self.threshold) -
                                     tf.reduce_mean(est_data_value), 0))

        # Variable
        dve_vars = [v for v in tf.trainable_variables()
                    if v.name.startswith('data_value_estimator')]

        # Solver
        dve_solver = tf.train.AdamOptimizer(self.learning_rate).minimize(
            dve_loss, var_list=dve_vars)

        LogUtil.log('INFO',"To evaluate x_valid with ori model!")
        # Baseline performance
        print(self.ori_model_path)
        y_valid_hat = eval_sgcn_prediction(self.x_valid, window=4, model_path=self.ori_model_path, \
                                           gpu_id=0, y_test=self.y_valid, predict_batch_size=self.batch_size_predictor)

        if perf_metric == 'auc':
            # valid_perf = metrics.roc_auc_score(self.y_valid, y_valid_hat[:, 1])
            valid_perf = metrics.roc_auc_score(self.y_valid_onehot, y_valid_hat)
        elif perf_metric == 'accuracy':
            valid_perf = metrics.accuracy_score(self.y_valid, np.argmax(y_valid_hat,
                                                                        axis=1))
        elif perf_metric == 'log_loss':
            valid_perf = -metrics.log_loss(self.y_valid, y_valid_hat)
        elif perf_metric == 'rmspe':
            valid_perf = rgvn_metrics.rmspe(self.y_valid, y_valid_hat)
        elif perf_metric == 'mae':
            valid_perf = metrics.mean_absolute_error(self.y_valid, y_valid_hat)
        elif perf_metric == 'mse':
            valid_perf = metrics.mean_squared_error(self.y_valid, y_valid_hat)

        LogUtil.log('INFO',"To evaluate x_train with val model!")
        # Prediction differences
        y_train_valid_pred = eval_sgcn_prediction(self.x_train, window=4, model_path=self.val_model_path, gpu_id=0,
                                                  y_test=self.y_train,
                                                  predict_batch_size=self.batch_size_predictor)

        if self.problem == 'classification':
            y_pred_diff = np.abs(self.y_train_onehot - y_train_valid_pred)
        elif self.problem == 'regression':
            y_pred_diff = \
                np.abs(self.y_train_onehot - y_train_valid_pred) / \
                self.y_train_onehot

        #Disable GPU Usage
        # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        # Main session
        session_conf = tf.ConfigProto(
            allow_soft_placement=False,
            log_device_placement=False)
        sess = tf.Session(config=session_conf)
        sess.run(tf.global_variables_initializer())
        # Model save at the end
        saver = tf.train.Saver(dve_vars)

        for _ in tqdm.tqdm(range(self.outer_iterations)):
            # Batch selection
            batch_idx = \
                np.random.permutation(len(self.x_train))[
                :self.batch_size]
            x_batch = self.x_train[batch_idx]
            y_batch_onehot = self.y_train_onehot[batch_idx]
            y_batch = self.y_train[batch_idx]
            y_hat_batch = y_pred_diff[batch_idx]

            LogUtil.log('INFO','hhhhhhhhhhhhhhhhhhhh')
            x_train_class = SGCNData(self.x_train, self.y_train, 4)
            alias_inputs, A, items, node_masks, targets = x_train_class.get_slice(batch_idx)

            LogUtil.log('INFO','Start to generate selection probability')
            # Generates selection probability
            print(x_batch)
            print(items)
            print(A)
            print(y_input)
            est_dv_curr = sess.run(
                est_data_value,
                feed_dict={
                    self.A: A,
                    # Liu Chenxu add
                    #self.x_input: x_batch,
                    self.items: items,
                    self.node_masks: node_masks,
                    self.y_input: y_batch_onehot,
                    self.y_hat_input: y_hat_batch
                })
            LogUtil.log('INFO','End to generate selection probability')
            # Samples the selection probability
            sel_prob_curr = np.random.binomial(
                1, est_dv_curr, est_dv_curr.shape)

            # Exception (When selection probability is 0)
            if np.sum(sel_prob_curr) == 0:
                est_dv_curr = 0.5 * np.ones(np.shape(est_dv_curr))
                sel_prob_curr = np.random.binomial(
                    1, est_dv_curr, est_dv_curr.shape)

            # Trains predictor
            flatten_sel_prob_curr = sel_prob_curr.flatten()
            weighted_x_batch = x_batch[np.where(flatten_sel_prob_curr > 0)]
            weighted_y_batch = y_batch[np.where(flatten_sel_prob_curr > 0)]
            LogUtil.log('INFO',"Start to train new model.")
            # new_model_batch_size = len(weighted_x_batch)
            new_model_path = train_sgcn(self.hidden_dim, self.label_dim, self.n_nodes, 1, weighted_x_batch,
                                        weighted_y_batch, 50, \
                                        'tmp/sgcn_as_predict_new_model', step_save_model=8, lr=0.001,
                                        epoch=self.inner_iterations)
            LogUtil.log('INFO',"New model training done.")
            LogUtil.log('INFO',new_model_path)
            # Prediction
            y_valid_hat = eval_sgcn_prediction(self.x_valid, window=4, model_path=new_model_path, \
                                               gpu_id=0, y_test=self.y_valid,
                                               predict_batch_size=self.batch_size_predictor)
            LogUtil.log('INFO',"Evaluate with new model done.")
            # Reward computation
            if perf_metric == 'auc':
                rgvn_perf = metrics.roc_auc_score(
                    # self.y_valid, y_valid_hat[:, 1])
                    self.y_valid_onehot, y_valid_hat)
            elif perf_metric == 'accuracy':
                rgvn_perf = metrics.accuracy_score(self.y_valid, np.argmax(y_valid_hat,
                                                                           axis=1))
            elif perf_metric == 'log_loss':
                rgvn_perf = -metrics.log_loss(self.y_valid, y_valid_hat)
            elif perf_metric == 'rmspe':
                rgvn_perf = rgvn_metrics.rmspe(self.y_valid, y_valid_hat)
            elif perf_metric == 'mae':
                rgvn_perf = metrics.mean_absolute_error(
                    self.y_valid, y_valid_hat)
            elif perf_metric == 'mse':
                rgvn_perf = metrics.mean_squared_error(
                    self.y_valid, y_valid_hat)

            if self.problem == 'classification':
                reward_curr = rgvn_perf - valid_perf
            elif self.problem == 'regression':
                reward_curr = valid_perf - rgvn_perf

            LogUtil.log('INFO','Start to train the generator')
            # Trains the generator
            _, _ = sess.run(
                [dve_solver, dve_loss],
                feed_dict={
                    self.A: A,
                    self.items: items,
                    self.node_masks: node_masks,
                    self.y_input: y_batch_onehot,
                    self.y_hat_input: y_hat_batch,
                    self.s_input: sel_prob_curr,
                    self.reward_input: reward_curr
                })
            LogUtil.log('INFO','End to train the generator')
        # Saves trained model
        saver.save(sess, self.checkpoint_file_name)
        LogUtil.log('INFO',"Saved trained rgvn model.")

    def data_valuator(self, x_train, y_train):
        """Returns data values using the data valuator model.

        Args:
          x_train: training features
          y_train: training labels

        Returns:
          final_dat_value: final data values of the training samples
        """

        # One-hot encoded labels
        if self.problem == 'classification':
            y_train_onehot = np.eye(len(np.unique(y_train)))[
                y_train.astype(int)]
            LogUtil.log('INFO',"Start to inference training data with valid model")
            y_train_valid_pred = eval_sgcn_prediction(x_train, window=4, model_path=self.val_model_path, \
                                                      gpu_id=0, y_test=y_train,
                                                      predict_batch_size=self.batch_size_predictor)
            # y_train_valid_pred = self.val_model.predict_proba(x_train)
        elif self.problem == 'regression':
            y_train_onehot = np.reshape(y_train, [len(y_train), 1])
            y_train_valid_pred = np.reshape(self.val_model.predict(x_train),
                                            [-1, 1])

        # Generates y_train_hat
        if self.problem == 'classification':
            y_train_hat = np.abs(y_train_onehot - y_train_valid_pred)
        elif self.problem == 'regression':
            y_train_hat = np.abs(
                y_train_onehot - y_train_valid_pred) / y_train_onehot

        # Restores the saved model
        LogUtil.log('INFO',"Restoring data evaluator.")
        imported_graph = \
            tf.train.import_meta_graph(self.checkpoint_file_name + '.meta')
        sess = tf.Session()
        imported_graph.restore(sess, self.checkpoint_file_name)
        est_data_value = self.rpm()
        # Estimates data value
        x_train_class = SGCNData(x_train, y_train, 4)
        batch_idx = list(range(len(x_train)))
        # Get full inference data
        slices = x_train_class.generate_batch(self.batch_size_predictor)
        LogUtil.log('INFO',"Start to inference with rgvn model")
        final_data_value = None
        for step in range(len(slices)):
            i = slices[step]
            alias_inputs, A, items, node_masks, targets = x_train_class.get_slice(i)
            targets_onehot = label_to_onehot(targets, self.label_dim)
            y_hat_batch = y_train_hat[i, :]
            # LogUtil.log('INFO','Flag test shape!')
            # LogUtil.log('INFO',targets.shape)
            # LogUtil.log('INFO',y_hat_batch.shape)
            batch_final_data_value = sess.run(
                est_data_value,
                feed_dict={
                    self.A: A,
                    self.items: items,
                    self.node_masks: node_masks,
                    self.y_input: targets_onehot,
                    self.y_hat_input: y_hat_batch})[:, 0]
            if step == 0:
                final_data_value = batch_final_data_value
            else:
                final_data_value = np.concatenate((final_data_value, batch_final_data_value))
        LogUtil.log('INFO',"End to inference with rgvn model")
        return final_data_value

    def rgvn_predictor(self, x_test):
        """Returns predictions using the predictor model.

        Args:
          x_test: testing features

        Returns:
          y_test_hat: predictions of the predictive model with DVRL
        """

        if self.flag_sgd:
            y_test_hat = self.final_model.predict(x_test)
        else:
            if self.problem == 'classification':
                y_test_hat = eval_sgcn_prediction(x_test, window=4, model_path=self.final_model_path, \
                                                  gpu_id=0, y_test=None, predict_batch_size=self.batch_size_predictor)
                # y_test_hat = self.final_model.predict_proba(x_test)
            elif self.problem == 'regression':
                y_test_hat = self.final_model.predict(x_test)

        return y_test_hat
