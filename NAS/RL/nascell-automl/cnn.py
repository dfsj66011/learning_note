# -*-coding: utf-8 -*-

import tensorflow as tf


class CNN:
    def __init__(self, num_input, num_classes, cnn_config):
        """
        Actions can modify filters:
            * the dimensionality of the output space,
            * kernel_size (integer, specifying the length of the 1D convolution window),
            * pool_size ( integer, representing the size of the pooling window)
            * dropout_rate per layer.
        """
        cnn = [c[0] for c in cnn_config]
        cnn_num_filters = [c[1] for c in cnn_config]
        max_pool_ksize = [c[2] for c in cnn_config]

        self.X = tf.placeholder(tf.float32, [None, num_input], name="input_X")   # [?, 784]
        self.Y = tf.placeholder(tf.int32, [None, num_classes], name="input_Y")   # [?, 10]
        self.dropout_keep_prob = tf.placeholder(tf.float32, [], name="dense_dropout_keep_prob")
        self.cnn_dropout_rates = tf.placeholder(tf.float32, [len(cnn), ], name="cnn_dropout_keep_prob")

        Y = self.Y
        X = tf.expand_dims(self.X, -1)
        pool_out = X
        with tf.name_scope("Conv_part"):
            for idd, filter_size in enumerate(cnn):
                with tf.name_scope("L"+str(idd)):
                    conv_out = tf.layers.conv1d(
                        pool_out,                            # [?, 784, 1]
                        filters=cnn_num_filters[idd],        # num_filter_1 / num_filter_2
                        kernel_size=(int(filter_size)),
                        strides=1,
                        padding="SAME",
                        name="conv_out_"+str(idd),
                        activation=tf.nn.relu,
                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                        bias_initializer=tf.zeros_initializer
                    )                                                   # [?, 784, num_filter_1 / num_filter_2]
                    pool_out = tf.layers.max_pooling1d(
                        conv_out,
                        pool_size=(int(max_pool_ksize[idd])),  # pool_size_1 / pool_size_2
                        strides=1,
                        padding='SAME',
                        name="max_pool_"+str(idd)
                    )                                                   # [?, 784, num_filter_1 / num_filter_2]
                    pool_out = tf.nn.dropout(pool_out, self.cnn_dropout_rates[idd])    # keep_prob >=1 不丢弃

            flatten_pred_out = tf.contrib.layers.flatten(pool_out)      # [?, 784*num_filter_2]
            self.logits = tf.layers.dense(flatten_pred_out, num_classes)   # [?, 10]

        self.prediction = tf.nn.softmax(self.logits, name="prediction")
        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=Y, name="loss")
        correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")
