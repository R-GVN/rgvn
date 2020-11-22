import csv
import datetime
import os
import time

import tensorflow as tf
import numpy as np

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

def train_textCNN2Input(x_train, x_1_train,y_train, vocab_processor, x_dev,x_1_dev, y_dev, parameter,gpu_id):
    path=None
    import tensorflow as tf
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=parameter["allow_soft_placement"],
          log_device_placement=parameter["log_device_placement"])
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN2Input(
                sequence_length=x_train.shape[1],
                sequence_length_1=x_1_train.shape[1],
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
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "textCNN2Inputs_runs", timestamp))
            print("Writing to {}\n".format(out_dir))

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

            def train_step(x_batch, x_1_batch,y_batch,dropout_keep_prob):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_x_1: x_1_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch,x_1_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_x_1: x_1_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)


            # Generate batches
            batches = batch_iter(
                list(zip(x_train,x_1_train, y_train)), parameter["batch_size"], parameter["num_epochs"])
            # Training loop. For each batch...
            for batch in batches:
                x_batch,x_1_batch, y_batch = zip(*batch)
                train_step(x_batch,x_1_batch, y_batch, parameter["dropout_keep_prob"])
                current_step = tf.train.global_step(sess, global_step)
                if current_step % parameter["evaluate_every"] == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev,x_1_dev, y_dev, writer=dev_summary_writer)
                    print("")
                if current_step % parameter["checkpoint_every"] == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
    return  path

def eval_textCNN2Input(x_raw,x_test,x_1_test,y_test,checkpoint_path,parameter):
    import tensorflow as tf
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_path)
    print("Prepare to load checkpoint from {}".format(checkpoint_file))
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
            input_x_1 = graph.get_operation_by_name("input_x_1").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = batch_iter(list(zip(x_test,x_1_test)), parameter["predict_batch_size"], 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []

            for x_test_batch in batches:
                x_batch,x_1_batch = zip(*x_test_batch)
                batch_predictions = sess.run(predictions, {input_x: x_batch,input_x_1:x_1_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

    # Print accuracy if y_test is defined
    if y_test is not None:
        correct_predictions = float(sum(all_predictions == y_test))
        print("Total number of test examples: {}".format(len(y_test)))
        print("Accuracy: {:g}".format(correct_predictions / float(len(y_test))))

    # Save the evaluation to a csv
    predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
    out_path = os.path.join(checkpoint_path, "..", "prediction.csv")
    print("Saving evaluation to {0}".format(out_path))
    with open(out_path, 'w') as f:
        csv.writer(f).writerows(predictions_human_readable)
    return all_predictions

class TextCNN2Input(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, sequence_length_1, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0,gpu=0):
        with tf.device('/device:GPU:%d' % gpu):
            # Placeholders for input, output and dropout
            self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
            self.input_x_1 = tf.placeholder(tf.int32, [None, sequence_length_1], name="input_x_1")
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
            self.embedded_chars_1 = tf.nn.embedding_lookup(self.W, self.input_x_1)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            self.embedded_chars_expanded_1 = tf.expand_dims(self.embedded_chars_1, -1)

            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            pooled_outputs_1 = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_size, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

                    W_1 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_1")
                    b_1 = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b_1")

                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")

                    conv_1 = tf.nn.conv2d(
                        self.embedded_chars_expanded_1,
                        W_1,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")

                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    h_1 = tf.nn.relu(tf.nn.bias_add(conv_1, b_1), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_1 = tf.nn.max_pool(
                        h_1,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)
                    pooled_outputs_1.append(pooled_1)

            # Combine all the pooled features
            num_filters_total = num_filters * len(filter_sizes)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_1 = tf.concat(pooled_outputs_1,3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
            self.h_pool_flat_1 = tf.reshape(self.h_pool_1, [-1, num_filters_total])

            # Add dropout
            with tf.name_scope("dropout"):
                self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
                self.h_drop_1 = tf.nn.dropout(self.h_pool_flat_1, self.dropout_keep_prob)

            self.h_drop = self.h_drop + self.h_drop_1

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

