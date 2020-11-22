import argparse
from utils import DataUtil,LogUtil
import data_loading
import rgvn_metrics
import numpy as np
def main(args):
    temp_data_path = "./metrics/temp_data_path"
    file1 = open(temp_data_path,"r")
    lines = file1.readlines()
    train_file_path = lines[0].strip()
    valid_file_path = lines[1].strip()
    test_file_path = lines[2].strip()

    temp_dve_out_path = "./metrics/data_value_out_temp.npy"
    dve_out = np.load(temp_dve_out_path)

    x_train, y_train, x_valid, y_valid, x_test, y_test, vocab_processor = \
        data_loading.process_data_for_textCNN(train_file_path, valid_file_path, test_file_path)
    LogUtil.log("INFO", "Start to evaluate textCNN on multiGPU")
    performance = rgvn_metrics.remove_high_low_textCNN(dve_out, x_train, y_train, \
                                                       x_valid, y_valid, x_test, y_test, vocab_processor, args.gpu_count,
                                                       'accuracy', plot=True)
    LogUtil.log("INFO", "End to evaluate textCNN on multiGPU")
    LogUtil.log('INFO',performance)
    np.savetxt("./metrics/TextCNN_performance_" + str(args.feature_method) + "_" + str(args.noise_rate),
               performance, fmt='%f', delimiter=',')
if __name__ == '__main__':

    # Inputs for the main function
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_name',
        choices=['adult', 'blog','ag_news'],
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
        '--gpu_count',
        help='Determine if to merge multiple column features into one column',
        default=4,
        type=int)
    args_in = parser.parse_args()

    # Calls main function
    main(args_in)
