import csv
import datetime
import os
import time

import gensim
import tensorflow as tf
import numpy as np
from utils import DataUtil,LogUtil

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def train_textCNN(x_train, y_train, vocab_processor, x_dev, y_dev, parameter,gpu_id=0):
    path=None
    import tensorflow as tf
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=parameter["allow_soft_placement"],
          log_device_placement=parameter["log_device_placement"])
        #session_conf.gpu_options.per_process_gpu_memory_fraction = 1
        session_conf.gpu_options.allow_growth = False
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=parameter["embedding_dim"],
                filter_sizes=list(map(int, parameter["filter_sizes"].split(","))),
                num_filters=parameter["num_filters"],
                l2_reg_lambda=parameter["l2_reg_lambda"],
                gpu=gpu_id)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(parameter["learning_rate"])
            # optimizer = tf.train.AdadeltaOptimizer(learning_rate=parameter["learning_rate"],rho=0.95, epsilon=1e-08)
            # optimizer = tf.train.MomentumOptimizer(learning_rate=parameter["learning_rate"], momentum=0.9)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp,"_"+str(gpu_id)))
            LogUtil.log('INFO',"Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=parameter["num_checkpoints"])

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "textCNN_vocab"))

            # Initialize all variables
            if "init_checkpoint" in parameter:
                sess.run(tf.train.init_from_checkpoint(parameter['init_checkpoint'],{'/':'/'}))
            else:
                sess.run(tf.global_variables_initializer())

            pretrain_word2vec = parameter['pretrain_word2vec']
            if pretrain_word2vec:
                LogUtil.log('INFO',"Loading a pretrained word2vec model")
                # initial matrix with random uniform
                initW = np.random.uniform(-0.25, 0.25, (len(vocab_processor.vocabulary_), parameter["embedding_dim"]))
                # Load pretrained word2vec model
                word_vecs = gensim.models.KeyedVectors.load(
                    pretrain_word2vec, mmap='r')

                ## Extract word:id mapping from the object.
                vocab_dict = vocab_processor.vocabulary_._mapping
                vocabulary = vocab_dict.values()

                for word in vocabulary:
                    if word in word_vecs:
                        idx = vocab_processor.vocabulary_.get(word)
                        initW[idx] = word_vecs[word]
                sess.run(cnn.W.assign(initW))
                LogUtil.log('INFO',"Loaded a pretrained word2vec model")

            def train_step(x_batch, y_batch,dropout_keep_prob):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                train_summary_writer.add_summary(summaries, step)
                return  step, loss, accuracy

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                #batches = batch_iter(list(x_batch), parameter["predict_batch_size"], 1, shuffle=False)

            # Collect the predictions here
                #all_predictions = []

                #for x_test_batch in batches:
                #   batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                #   all_predictions = np.concatenate([all_predictions, batch_predictions])
                #correct_predictions = float(sum(all_predictions == y_batch))
                #LogUtil.log('INFO',"Total number of test examples: {}".format(len(y_batch)))
                #LogUtil.log('INFO',"Accuracy: {:g}".format(correct_predictions / float(len(y_batch))))
                #accuracy = correct_predictions / float(len(y_batch))
                #loss = 0
                batches = batch_iter(list(zip(x_batch, y_batch)), parameter["batch_size"], 1, False)

                #x_batch = x_batch[:64]
                #y_batch = y_batch[:64]
                all_accuracy = 0.0
                all_loss = 0.0
                count = 0.0
                for batch in batches:
                    x_train, y_train = zip(*batch)
                    feed_dict = {
                      cnn.input_x: x_train,
                      cnn.input_y: y_train,
                      cnn.dropout_keep_prob: 1.0
                    }
                    step, summaries, loss, accuracy = sess.run(
                        [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                        feed_dict)
                    all_accuracy += accuracy
                    all_loss += loss
                    count+=1
                    if count>20:
                        break
                accuracy = all_accuracy/count
                loss = all_loss/count

                time_str = datetime.datetime.now().isoformat()
                LogUtil.log('INFO',"{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)
                return loss, accuracy


            # Generate batches
            batches = batch_iter(
                list(zip(x_train, y_train)), parameter["batch_size"], parameter["num_epochs"])
            # Training loop. For each batch...
            best_eval_accuracy = 0
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                step, loss, accuracy = train_step(x_batch, y_batch, parameter["dropout_keep_prob"])
                current_step = tf.train.global_step(sess, global_step)
                if current_step % parameter["evaluate_every"] == 0:
                    time_str = datetime.datetime.now().isoformat()
                    LogUtil.log('INFO',"{}: Training step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    LogUtil.log('INFO',"\nEvaluation:")

                    # the batch size of x_dev is too large
                    eval_loss,eval_acc = dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    if eval_acc > best_eval_accuracy:
                        best_eval_accuracy = eval_acc
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        LogUtil.log('INFO',"Saved model checkpoint to {}\n".format(path))
                # if current_step % parameter["checkpoint_every"] == 0 and accuracy > best_accuracy:
                #     best_accuracy = accuracy
                #     path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                #     LogUtil.log('INFO',"Saved model checkpoint to {}\n".format(path))
    return  path

def eval_textCNN(x_raw,x_test,y_test,checkpoint_path,parameter):
    import tensorflow as tf
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_path)
    LogUtil.log('INFO',"Prepare to load checkpoint from {}".format(checkpoint_file))
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=parameter["allow_soft_placement"],
            log_device_placement=parameter["log_device_placement"])
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = batch_iter(list(x_test), parameter["predict_batch_size"], 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []

            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

    # Print accuracy if y_test is defined
    if y_test is not None:
        correct_predictions = float(sum(all_predictions == y_test))
        LogUtil.log('INFO',"Total number of test examples: {}".format(len(y_test)))
        LogUtil.log('INFO',"Accuracy: {:g}".format(correct_predictions / float(len(y_test))))

    if x_raw is not None:
        # Save the evaluation to a csv
        predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
        out_path = os.path.join(checkpoint_path, "..", "prediction.csv")
        LogUtil.log('INFO',"Saving evaluation to {0}".format(out_path))
        with open(out_path, 'w') as f:
            csv.writer(f).writerows(predictions_human_readable)
    return all_predictions

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0,gpu=0):
        # Placeholders for input, output and dropout
        with tf.device('/device:GPU:%d' % gpu):
            self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

            # Keeping track of l2 regularization loss (optional)
            l2_loss = tf.constant(0.0)

            # Embedding layer
            # with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_size, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)

            # Combine all the pooled features
            num_filters_total = num_filters * len(filter_sizes)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

            # Add dropout
            with tf.name_scope("dropout"):
                self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

            # Final (unnormalized) scores and predictions
            with tf.name_scope("output"):
                W = tf.get_variable(
                    "W",
                    shape=[num_filters_total, num_classes],
                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
                self.predictions = tf.argmax(self.scores, 1, name="predictions")

            # Calculate mean cross-entropy loss
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
                self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

            # Accuracy
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

