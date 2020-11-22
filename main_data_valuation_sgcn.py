from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import os
from utils import DataUtil,LogUtil
# import keras
import lightgbm
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import shutil
import data_loading
import rgvn_sgcn
import rgvn_metrics
from sgcn_utils import SGCN

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
def main(args):
    """Main function of RGVN for data valuation experiment.

    Args:
      args: data_name, train_no, valid_no,
            normalization, network parameters, number of examples
    """
    # Data loading and sample corruption
    data_name = args.data_name

    # The number of training and validation samples
    dict_no = dict()
    dict_no['train'] = args.train_no
    dict_no['valid'] = args.valid_no

    # Network parameters
    parameters = dict()
    parameters['hidden_dim'] = args.hidden_dim
    parameters['comb_dim'] = args.comb_dim
    parameters['iterations'] = args.iterations
    parameters['activation'] = tf.nn.relu
    parameters['inner_iterations'] = args.inner_iterations
    parameters['layer_number'] = args.layer_number
    parameters['learning_rate'] = args.learning_rate
    parameters['batch_size'] = args.batch_size
    parameters['batch_size_predictor'] = args.batch_size_predictor
    parameters['n_label_class'] = args.n_label_class

    # The number of examples
    n_exp = args.n_exp

    # Checkpoint file name
    checkpoint_file_name = args.checkpoint_file_name

    # Resevered for accelerating code debug
    # Data loading
    noise_idx,vocab_count, train_file_path,valid_file_path,test_file_path = data_loading.load_tabular_data(\
        data_name, dict_no, args.noise_rate, args.vocab_size, args.feature_method,args.is_merge_col)
    LogUtil.log('INFO','Finished data loading.')
    #
    temp_data_path = "./metrics/temp_data_path"
    if os.path.exists(temp_data_path):
        os.remove(temp_data_path)
    with open(temp_data_path,"w") as fw:
        fw.write(train_file_path+"\n")
        fw.write(valid_file_path+"\n")
        fw.write(test_file_path+"\n")
    #
    # # Data preprocessing
    # # Normalization methods: 'minmax' or 'standard'
    normalization = args.normalization
    # # Extracts features and labels. Then, normalizes features
    x_train, y_train, x_valid, y_valid, x_test, y_test, col_names = \
            data_loading.preprocess_data(args.data_name,normalization, train_file_path \
                                         ,valid_file_path,test_file_path)

    if args.feature_method == 'sgcn' :
        LogUtil.log('INFO',str(len(x_train)))
    else:
        LogUtil.log('INFO',str(len(x_train[0, :])))
    LogUtil.log('INFO','Finished data preprocess.')

    # Run DVRL
    # Resets the graph
    tf.reset_default_graph()
    # keras.backend.clear_session()

    # Here, we assume a classification problem and we assume a predictor model
    # in the form of a simple multi-layer perceptron.
    problem = 'classification'
    # Predictive model define
    if args.feature_method == "sgcn":
        pred_model =SGCN(parameters['hidden_dim'], vocab_count, parameters['n_label_class'],1,0)
    else:
        pred_model = keras.models.Sequential()
        # if args.feature_method != "sgcn":
        pred_model.add(keras.layers.Dense(parameters['hidden_dim'],
                                          activation='relu'))
        pred_model.add(keras.layers.Dense(parameters['hidden_dim'],
                                          activation='relu'))
        pred_model.add(keras.layers.Dense(parameters['n_label_class'], activation='softmax'))
        pred_model.compile(optimizer='adam', loss='categorical_crossentropy',
                           metrics=['accuracy'])
    # Flags for using stochastic gradient descent / pre-trained model
    flags = {'sgd': True, 'pretrain': False}
    #Remove files in tmp before rgvn training
    if os.path.exists('./tmp'):
        filelist = [f for f in os.listdir('./tmp')]
        for f in filelist:
            os.remove(os.path.join('./tmp', f))
    checkpoint_dir = os.path.dirname(checkpoint_file_name)
    if os.path.exists(checkpoint_dir):
        filelist = [f for f in os.listdir(checkpoint_dir)]
        for f in filelist:
            os.remove(os.path.join(checkpoint_dir, f))
    # Initializes DVRL
    rgvn_class = rgvn_sgcn.RgvnSGCN(vocab_count, x_train, y_train, x_valid, y_valid,
                           problem, pred_model, parameters,
                           checkpoint_file_name, flags)
    LogUtil.log('INFO',"Init rgvn succeed!")
    rgvn_class.train_rgvn('accuracy')
    LogUtil.log('INFO','Finished rgvn training.')

    # Outputs
    # Data valuation
    dve_out = rgvn_class.data_valuator(x_train, y_train)
    if not os.path.exists('metrics'):
        os.makedirs('metrics')
    np.savetxt("./metrics/data_value_out_"+str(args.feature_method)+"_"+str(args.noise_rate),np.array(dve_out),fmt='%f', delimiter=',')
    temp_dve_out_path = "./metrics/data_value_out_temp"
    if os.path.exists(temp_dve_out_path):
        os.remove(temp_dve_out_path)
    np.save(temp_dve_out_path,np.array(dve_out))
    LogUtil.log('INFO','Finished data valuation.')

    # Evaluations
    # 1. Data valuation
    # Data valuation
    sorted_idx = np.argsort(-dve_out)
    sorted_x_train = x_train[sorted_idx]

    # Indices of top n high valued samples
    LogUtil.log('INFO','Indices of top ' + str(n_exp) + ' high valued samples: '
          + str(sorted_idx[:n_exp]))
    LogUtil.log('INFO',pd.DataFrame(data=sorted_x_train[:n_exp], index=range(n_exp),
                       columns=col_names).head())

    # Indices of top n low valued samples
    LogUtil.log('INFO','Indices of top ' + str(n_exp) + ' low valued samples: '
          + str(sorted_idx[-n_exp:]))
    LogUtil.log('INFO',pd.DataFrame(data=sorted_x_train[-n_exp:], index=range(n_exp),
                       columns=col_names).head())
    np.savetxt("./metrics/lowest_value_out_"+str(args.feature_method)+"_"+str(args.noise_rate),np.array(sorted_idx[-n_exp:]),fmt='%d', delimiter=',')
    np.savetxt("./metrics/noisy_index_"+str(args.feature_method)+"_"+str(args.noise_rate),np.array(noise_idx),fmt='%f', delimiter=',')

    # 2. Performance after removing high/low values
    # Here, as the evaluation model, we use LightGBM.
    if args.eval_model.lower() == "lightgbm":
        eval_model = lightgbm.LGBMClassifier()
        eval_model.set_params(objective="multiclass")
        eval_model.set_params(num_class=args.n_label_class)
        LogUtil.log("INFO","Start to evaluate with lightgbm!")
        # Performance after removing high/low values
        performance = rgvn_metrics.remove_high_low(dve_out, eval_model, x_train, y_train,
                                         x_valid, y_valid, x_test, y_test,
                                         'accuracy', plot=True)
        LogUtil.log("INFO","End to evaluate with lightgbm!")
        LogUtil.log('INFO',performance)
        np.savetxt("./metrics/lightGBM_performance_"+str(args.feature_method)+"_"+str(args.noise_rate),performance,fmt='%f', delimiter=',')
    elif args.eval_model.lower() == "textcnn":
        LogUtil.log('INFO',"Start to evaluate with TextCNN!")
        x_train,y_train,x_valid,y_valid,x_test,y_test,vocab_processor = \
            data_loading.process_data_for_textCNN(train_file_path,valid_file_path,test_file_path)
        performance = rgvn_metrics.remove_high_low_textCNN(dve_out, x_train, y_train,\
                                         x_valid, y_valid, x_test, y_test,vocab_processor,args.gpu_count,
                                         'accuracy', plot=True)
        LogUtil.log('INFO',"End to evaluate with TextCNN!")
        LogUtil.log('INFO',performance)
        np.savetxt("./metrics/TextCNN_performance_E2EWordEmbedding_"+str(args.feature_method)+"_"+str(args.noise_rate),performance,fmt='%f', delimiter=',')

if __name__ == '__main__':

    # Inputs for the main function
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_name',
        choices=['adult', '20ng','ag_news'],
        help='data name (adult or blog or ag_news)',
        default='adult',
        type=str)
    parser.add_argument(
        '--normalization',
        choices=['minmax', 'standard','none'],
        help='data normalization method',
        default='minmax',
        type=str)
    parser.add_argument(
        '--eval_model',
        help='The model used for evaluating DVRL performance.',
        default='lightGBM',
        type=str)
    parser.add_argument(
        '--feature_method',
        help='The algo used to generate feature for raw texts',
        default='word2vec',
        type=str)
    parser.add_argument(
        '--train_no',
        help='number of training samples',
        default=1000,
        type=int)
    parser.add_argument(
        '--valid_no',
        help='number of validation samples',
        default=400,
        type=int)
    parser.add_argument(
        '--hidden_dim',
        help='dimensions of hidden states',
        default=100,
        type=int)
    parser.add_argument(
        '--comb_dim',
        help='dimensions of hidden states after combinding with prediction diff',
        default=10,
        type=int)
    parser.add_argument(
        '--layer_number',
        help='number of network layers',
        default=5,
        type=int)
    parser.add_argument(
        '--noise_rate',
        help='noise rate to generate the noisy label',
        default=0.0,
        type=float)
    parser.add_argument(
        '--iterations',
        help='number of iterations',
        default=2000,
        type=int)
    parser.add_argument(
        '--batch_size',
        help='number of batch size for RL',
        default=2000,
        type=int)
    parser.add_argument(
        '--inner_iterations',
        help='number of iterations',
        default=100,
        type=int)
    parser.add_argument(
        '--batch_size_predictor',
        help='number of batch size for predictor',
        default=256,
        type=int)
    parser.add_argument(
        '--n_exp',
        help='number of examples',
        default=5,
        type=int)
    parser.add_argument(
        '--vocab_size',
        help='vocab size for BOW embedding',
        default=800,
        type=int)
    parser.add_argument(
        '--n_label_class',
        help='number of class of label',
        default=2,
        type=int)
    parser.add_argument(
        '--learning_rate',
        help='learning rates for RL',
        default=0.01,
        type=float)
    parser.add_argument(
        '--checkpoint_file_name',
        help='file name for saving and loading the trained model',
        default='./tmp/model.ckpt',
        type=str)
    parser.add_argument(
        '--is_merge_col',
        help='Determine if to merge multiple column features into one column',
        default=1,
        type=int)
    parser.add_argument(
        '--gpu_count',
        help='Determine if to merge multiple column features into one column',
        default=4,
        type=int)
    args_in = parser.parse_args()

    # Calls main function
    main(args_in)
