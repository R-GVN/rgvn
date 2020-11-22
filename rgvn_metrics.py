from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import metrics
from text_cnn import train_textCNN, eval_textCNN
from text_cnn_2input import train_textCNN2Input, eval_textCNN2Input
from utils import DataUtil,LogUtil

def rmspe(data_y, pred_y):
    """Computes Root Mean Squared Percentage Error (RMSPE).

    Args:
      data_y: ground truth labels
      pred_y: predicted labels

    Returns:
      output_perf: RMSPE performance
    """
    data_y = np.reshape(np.asarray(data_y), [len(data_y), ])
    pred_y = np.reshape(pred_y, [len(pred_y), ])

    output_perf = np.sqrt(np.mean(((data_y - pred_y)/data_y)**2))

    return output_perf


def auroc(data_y, pred_y):
    """Computes Area Under ROC curve (AUROC) with reshaping.

    Args:
      data_y: ground truth labels
      pred_y: predicted labels

    Returns:
      output_perf: AUROC performance
    """
    data_y = np.reshape(np.asarray(data_y), [len(data_y), ])
    pred_y = np.reshape(pred_y, [len(pred_y), ])

    output_perf = metrics.roc_auc_score(data_y, pred_y)

    return output_perf


def discover_corrupted_sample(dve_out, noise_idx, noise_rate, plot=True):
    """Reports True Positive Rate (TPR) of corrupted label discovery.

    Args:
      dve_out: data values
      noise_idx: noise index
      noise_rate: the ratio of noisy samples
      plot: print plot or not

    Returns:
      output_perf: True positive rate (TPR) of corrupted label discovery
                   (per 5 percentiles)
    """

    # Sorts samples by data values
    num_bins = 20  # Per 100/20 percentile
    sort_idx = np.argsort(dve_out)

    # Output initialization
    output_perf = np.zeros([num_bins, ])

    # For each percentile
    for itt in range(num_bins):
        # from low to high data values
        output_perf[itt] = len(np.intersect1d(sort_idx[:int((itt+1) *
                                                            len(dve_out)/num_bins)], noise_idx)) \
            / len(noise_idx)

    # Plot corrupted label discovery graphs
    if plot:

        # Defines x-axis
        num_x = int(num_bins/2 + 1)
        x = [a*(1.0/num_bins) for a in range(num_x)]

        # Corrupted label discovery results (rgvn, optimal, random)
        y_rgvn = np.concatenate((np.zeros(1), output_perf[:(num_x-1)]))
        y_opt = [min([a*((1.0/num_bins)/noise_rate), 1]) for a in range(num_x)]
        y_random = x

        plt.figure(figsize=(6, 7.5))
        plt.plot(x, y_rgvn, 'o-')
        plt.plot(x, y_opt, '--')
        plt.plot(x, y_random, ':')
        plt.xlabel('Fraction of data Inspected', size=16)
        plt.ylabel('Fraction of discovered corrupted samples', size=16)
        plt.legend(['DVRL', 'Optimal', 'Random'], prop={'size': 16})
        plt.title('Corrupted Sample Discovery', size=16)
        plt.show()

    # Returns True Positive Rate of corrupted label discovery
    return output_perf

def remove_high_low_textCNN(dve_out, x_train, y_train,
                    x_valid, y_valid, x_test, y_test,vocab_processor,gpu_count = 4,
                    perf_metric='acc', plot=True):
    """Evaluates performance after removing a portion of high/low valued samples.

    Args:
      dve_out: data values
      x_train: training features
      y_train: training labels
      x_valid: validation features
      y_valid: validation labels
      x_test: testing features
      y_test: testing labels
      perf_metric: 'auc', 'accuracy', or 'rmspe'
      plot: print plot or not

    Returns:
      output_perf: Prediction performances after removing a portion of high
                   or low valued samples.
    """
    base_parameter = {
        'allow_soft_placement': False,
        'log_device_placement': False,
        'embedding_dim':300,
        'filter_sizes':"3,4,5",
        'num_filters':256,
        'l2_reg_lambda': 0.1,
        'num_checkpoints':3,
        'batch_size':64,
        'num_epochs':10,
        #Dropout keep probability (default: 0.5)
        'dropout_keep_prob':0.5,
        #Save model after this many steps(default: 100)
        'checkpoint_every':100,
        #Evaluate model on dev set after this many steps
        'evaluate_every':100,
        'predict_batch_size':64,
        'learning_rate':0.001,
        'pretrain_word2vec':'../data/GoogleNews_vectors-negative300_uncompress_noL2Norm'
    }

    # Sorts samples by data values
    num_bins = 20  # Per 100/20 percentile
    sort_idx = np.argsort(dve_out)
    n_sort_idx = np.argsort(-dve_out)

    # Output Initialization
    if perf_metric in ['auc', 'accuracy']:
        temp_output = np.zeros([2 * num_bins, 2])
    elif perf_metric == 'rmspe':
        temp_output = np.ones([2 * num_bins, 2])

    #Remove models before training
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs"))
    DataUtil.delete_files(out_dir)
    if gpu_count >= 1:
        LogUtil.log("INFO","Start multiprocessing with multi GPU")
        def train_and_test_textCNN(x_train, y_train, vocab_processor, x_valid, y_valid,x_test,y_test, parameter,gpu,return_dict):
            # with tf.device('/device:GPU:%d' % gpu):
            #os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu)
            import tensorflow as tf
            LogUtil.log("INFO","Start training with GPU "+str(gpu))
            model_path = train_textCNN(x_train, y_train, vocab_processor, x_valid, y_valid, parameter,gpu)
            model_dir = os.path.dirname(model_path)
            LogUtil.log("INFO","End training with GPU "+str(gpu))

            y_test_hat = eval_textCNN(None, x_test, None, model_dir, parameter)
            y_valid_hat = eval_textCNN(None, x_valid, None, model_dir, parameter)

            valid_acc = metrics.accuracy_score(np.argmax(y_valid, axis=1), y_valid_hat)
            test_acc = metrics.accuracy_score(y_test, y_test_hat)

            # print("Flag0"+str(gpu)+str(test_acc))
            return_dict[gpu] = valid_acc,test_acc
            LogUtil.log("INFO", "End evaluating with GPU " + str(gpu))

        tf.debugging.set_log_device_placement(True)
        iter_times = int(num_bins/gpu_count)
        for iter in range(iter_times):
            process_list = list()
            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            #Remove least value first
            for gpu in range(gpu_count):
                parameter = base_parameter
                itt = iter * gpu_count + gpu
                new_x_train = x_train[sort_idx[int(
                    itt * len(x_train[:, 0]) / num_bins):], :]
                new_y_train = y_train[sort_idx[int(itt * len(x_train[:, 0]) / num_bins):]]
                base_evaluate_every = base_parameter['evaluate_every']
                max_step = int(len(new_x_train) * base_parameter['num_epochs']/base_parameter['batch_size'])
                if base_evaluate_every > max_step:
                    parameter['evaluate_every'] = max_step
                process = multiprocessing.Process(target=train_and_test_textCNN,args=(new_x_train, new_y_train, vocab_processor, x_valid,
                                          y_valid,x_test,y_test, parameter,gpu,return_dict))
                process_list.append(process)
                process.start()
            for process in process_list:
                process.join()
            for gpu in range(gpu_count):
                temp_output[iter * gpu_count + gpu, 0] = return_dict[gpu][0]
                temp_output[iter * gpu_count + gpu, 1]= return_dict[gpu][1]

            process_list.clear()
            return_dict.clear()

            #Remove most valuable first
            for gpu in range(gpu_count):
                parameter = base_parameter
                itt = iter * gpu_count + gpu

                new_x_train = x_train[n_sort_idx[int(
                    itt * len(x_train[:, 0]) / num_bins):], :]
                new_y_train = y_train[n_sort_idx[int(
                    itt * len(x_train[:, 0]) / num_bins):]]
                base_evaluate_every = base_parameter['evaluate_every']
                max_step = int(len(new_x_train) * base_parameter['num_epochs']/base_parameter['batch_size'])
                if base_evaluate_every > max_step:
                    parameter['evaluate_every'] = max_step
                process = multiprocessing.Process(target=train_and_test_textCNN,args=(new_x_train, new_y_train, vocab_processor, x_valid,
                                          y_valid,x_test,y_test, parameter,gpu,return_dict))
                process_list.append(process)
                process.start()
            for process in process_list:
                process.join()
            for gpu in range(gpu_count):
                temp_output[iter * gpu_count + gpu + num_bins, 0] = return_dict[gpu][0]
                temp_output[iter * gpu_count + gpu + num_bins, 1]= return_dict[gpu][1]
        LogUtil.log("INFO", "End multiprocessing with multi GPU")
    else:
        # For each percentile bin
        #for itt in range(int(num_bins/2)):
        # split 3 step because multiprocess has some problem
        # run on 3 gpu by hand run 3 times
        for itt in range(0,3):
        #for itt in range(3,6):
        #for itt in range(6,10):
            LogUtil.log("INFO","Start to evaluate with TextCNN by removing "+str(itt)+ " part of training data")
            # 1. Remove least valuable samples first
            new_x_train = x_train[sort_idx[int(
                itt*len(x_train[:, 0])/num_bins):], :]
            new_y_train = y_train[sort_idx[int(itt*len(x_train[:, 0])/num_bins):]]

            if len(np.unique(new_y_train)) > 1:
                model_path = train_textCNN(new_x_train,new_y_train,vocab_processor,x_valid,y_valid,base_parameter)
                model_dir = os.path.dirname(model_path)
                y_test_hat = eval_textCNN(None, x_test, None, model_dir, base_parameter)
                #y_valid_hat = eval_textCNN(None, x_valid, None, model_dir, base_parameter)
                y_valid_hat = y_test_hat
                temp_output[itt, 0] = metrics.accuracy_score(np.argmax(y_valid,axis=1),y_valid_hat)
                #temp_output[itt, 1] = metrics.accuracy_score(y_test,y_test_hat)
                LogUtil.log("INFO", "remove least itt:{}, acc:{}".format(itt, temp_output[itt, 0]))

            # 2. Remove most valuable samples first
            new_x_train = x_train[n_sort_idx[int(
                itt*len(x_train[:, 0])/num_bins):], :]
            new_y_train = y_train[n_sort_idx[int(
                itt*len(x_train[:, 0])/num_bins):]]

            if len(np.unique(new_y_train)) > 1:
                model_path = train_textCNN(new_x_train, new_y_train, vocab_processor, x_valid, y_valid, base_parameter)
                model_dir = os.path.dirname(model_path)
                y_test_hat = eval_textCNN(None, x_test, None, model_dir, base_parameter)
                y_valid_hat = y_test_hat
                #y_valid_hat = eval_textCNN(None, x_valid, None, model_dir, base_parameter)

                temp_output[num_bins + itt, 0] = metrics.accuracy_score(np.argmax(y_valid,axis=1), y_valid_hat)
                #temp_output[num_bins + itt, 1] = metrics.accuracy_score(y_test, y_test_hat)
                LogUtil.log("INFO", "remove most itt:{}, acc:{}".format(itt, temp_output[num_bins + itt, 0]))
    # Plot graphs
    if plot:
        # Defines x-axis
        num_x = int(num_bins/2 + 1)
        x = [a*(1.0/num_bins) for a in range(num_x)]

        # Prediction performances after removing high or low values
        plt.figure(figsize=(6, 7.5))
        plt.plot(x, temp_output[:num_x, 1], 'o-')
        plt.plot(x, temp_output[num_bins:(num_bins+num_x), 1], 'x-')

        plt.xlabel('Fraction of Removed Samples', size=16)
        plt.ylabel('Accuracy', size=16)
        plt.legend(['Removing low value data', 'Removing high value data'],
                   prop={'size': 16})
        plt.title('Remove High/Low Valued Samples', size=16)

        plt.show()

    return temp_output

def remove_high_low(dve_out, eval_model, x_train, y_train,
                    x_valid, y_valid, x_test, y_test,
                    perf_metric='rmspe', plot=True):
    """Evaluates performance after removing a portion of high/low valued samples.

    Args:
      dve_out: data values
      eval_model: evaluation model (object)
      x_train: training features
      y_train: training labels
      x_valid: validation features
      y_valid: validation labels
      x_test: testing features
      y_test: testing labels
      perf_metric: 'auc', 'accuracy', or 'rmspe'
      plot: print plot or not

    Returns:
      output_perf: Prediction performances after removing a portion of high
                   or low valued samples.
    """

    x_train = np.asarray(x_train)
    y_train = np.reshape(np.asarray(y_train), [len(y_train), ])
    x_valid = np.asarray(x_valid)
    y_valid = np.reshape(np.asarray(y_valid), [len(y_valid), ])
    x_test = np.asarray(x_test)
    y_test = np.reshape(np.asarray(y_test), [len(y_test), ])

    # Sorts samples by data values
    num_bins = 20  # Per 100/20 percentile
    sort_idx = np.argsort(dve_out)
    n_sort_idx = np.argsort(-dve_out)

    # Output Initialization
    if perf_metric in ['auc', 'accuracy']:
        temp_output = np.zeros([2 * num_bins, 2])
    elif perf_metric == 'rmspe':
        temp_output = np.ones([2 * num_bins, 2])

    # For each percentile bin
    for itt in range(num_bins):

        # 1. Remove least valuable samples first
        new_x_train = x_train[sort_idx[int(
            itt*len(x_train[:, 0])/num_bins):], :]
        new_y_train = y_train[sort_idx[int(itt*len(x_train[:, 0])/num_bins):]]

        if len(np.unique(new_y_train)) > 1:

            eval_model.fit(new_x_train, new_y_train)

            if perf_metric == 'auc':
                y_valid_hat = eval_model.predict_proba(x_valid)[:, 1]
                y_test_hat = eval_model.predict_proba(x_test)[:, 1]

                temp_output[itt, 0] = auroc(y_valid, y_valid_hat)
                temp_output[itt, 1] = auroc(y_test, y_test_hat)

            elif perf_metric == 'accuracy':
                y_valid_hat = eval_model.predict_proba(x_valid)
                y_test_hat = eval_model.predict_proba(x_test)

                temp_output[itt, 0] = metrics.accuracy_score(y_valid,
                                                             np.argmax(y_valid_hat,
                                                                       axis=1))
                temp_output[itt, 1] = metrics.accuracy_score(y_test,
                                                             np.argmax(y_test_hat,
                                                                       axis=1))
            elif perf_metric == 'rmspe':
                y_valid_hat = eval_model.predict(x_valid)
                y_test_hat = eval_model.predict(x_test)

                temp_output[itt, 0] = rmspe(y_valid, y_valid_hat)
                temp_output[itt, 1] = rmspe(y_test, y_test_hat)

        # 2. Remove most valuable samples first
        new_x_train = x_train[n_sort_idx[int(
            itt*len(x_train[:, 0])/num_bins):], :]
        new_y_train = y_train[n_sort_idx[int(
            itt*len(x_train[:, 0])/num_bins):]]

        if len(np.unique(new_y_train)) > 1:

            eval_model.fit(new_x_train, new_y_train)

            if perf_metric == 'auc':
                y_valid_hat = eval_model.predict_proba(x_valid)[:, 1]
                y_test_hat = eval_model.predict_proba(x_test)[:, 1]

                temp_output[num_bins + itt, 0] = auroc(y_valid, y_valid_hat)
                temp_output[num_bins + itt, 1] = auroc(y_test, y_test_hat)

            elif perf_metric == 'accuracy':
                y_valid_hat = eval_model.predict_proba(x_valid)
                y_test_hat = eval_model.predict_proba(x_test)

                temp_output[num_bins + itt, 0] = \
                    metrics.accuracy_score(
                        y_valid, np.argmax(y_valid_hat, axis=1))
                temp_output[num_bins + itt, 1] = \
                    metrics.accuracy_score(
                        y_test, np.argmax(y_test_hat, axis=1))

            elif perf_metric == 'rmspe':
                y_valid_hat = eval_model.predict(x_valid)
                y_test_hat = eval_model.predict(x_test)

                temp_output[num_bins + itt, 0] = rmspe(y_valid, y_valid_hat)
                temp_output[num_bins + itt, 1] = rmspe(y_test, y_test_hat)

    # Plot graphs
    if plot:

        # Defines x-axis
        num_x = int(num_bins/2 + 1)
        x = [a*(1.0/num_bins) for a in range(num_x)]

        # Prediction performances after removing high or low values
        plt.figure(figsize=(6, 7.5))
        plt.plot(x, temp_output[:num_x, 1], 'o-')
        plt.plot(x, temp_output[num_bins:(num_bins+num_x), 1], 'x-')

        plt.xlabel('Fraction of Removed Samples', size=16)
        plt.ylabel('Accuracy', size=16)
        plt.legend(['Removing low value data', 'Removing high value data'],
                   prop={'size': 16})
        plt.title('Remove High/Low Valued Samples', size=16)

        plt.show()

    return temp_output


def learn_with_rgvn(dve_out, eval_model, x_train, y_train, x_valid, y_valid,
                    x_test, y_test, perf_metric):
    """Evalautes performance of the model with DVRL training.

    Args:
      dve_out: data values
      eval_model: evaluation model (object)
      x_train: training features
      y_train: training labels
      x_valid: validation features
      y_valid: validation labels
      x_test: testing features
      y_test: testing labels
      perf_metric: 'auc', 'accuracy', or 'rmspe'

    Returns:
      output_perf: Prediction performances of evaluation model with rgvn.
    """
    temp_output = \
        remove_high_low(dve_out, eval_model, x_train, y_train,
                        x_valid, y_valid, x_test, y_test,
                        perf_metric, plot=False)

    if perf_metric == 'rmspe':
        opt_itt = np.argmin(temp_output[:, 0])
    elif perf_metric in ['auc', 'accuracy']:
        opt_itt = np.argmax(temp_output[:, 0])

    output_perf = temp_output[opt_itt, 1]

    return output_perf


def learn_with_baseline(eval_model, x_train, y_train, x_test, y_test,
                        perf_metric):
    """Evaluates performance of baseline evaluation model without data valuation.

    Args:
      eval_model: evaluation model (object)
      x_train: training features
      y_train: training labels
      x_test: testing features
      y_test: testing labels
      perf_metric: 'auc', 'accuracy', or 'rmspe'

    Returns:
      output_perf: Prediction performance of baseline predictive model
    """
    eval_model.fit(x_train, y_train)

    if perf_metric == 'auc':
        y_test_hat = eval_model.predict_proba(x_test)[:, 1]
        output_perf = auroc(y_test, y_test_hat)
    elif perf_metric == 'accuracy':
        y_test_hat = eval_model.predict_proba(x_test)
        output_perf = metrics.accuracy_score(
            y_test, np.argmax(y_test_hat, axis=1))
    elif perf_metric == 'rmspe':
        y_test_hat = eval_model.predict(x_test)
        output_perf = rmspe(y_test, y_test_hat)

    return output_perf
