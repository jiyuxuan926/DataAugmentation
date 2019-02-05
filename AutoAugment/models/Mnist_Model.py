import time
import numpy as np
import tensorflow as tf

from data.mnist import MNIST
from models.AbstractModel import AbstractModel, create_batches, random_batch

class MNISTModel(AbstractModel):
    def __init__(self, MAXIT = 500, BATCH_SIZE = 256, lr = 1e-2):
        self.MAXIT = MAXIT
        self.BATCH_SIZE = BATCH_SIZE
        self.lr = lr

        self.model_sess = tf.Session()
        self.x = tf.placeholder(tf.float64, [None, 28, 28, 3], name = "Images")
        self.y = tf.placeholder(tf.int64  , [None, 10], name = 'Labels')
        self.build_model()
        self.model_sess.run(tf.global_variables_initializer())
        super().__init__()

    def build_model(self):
        """
        Implements the tensorflow computational graph for the child model to be trained
        """
        tf.reset_default_graph()
        print("Building model...")
        with tf.name_scope('Conv1'):
            w_conv1 = tf.get_variable("conv1_weights", shape=[5, 5, 3, 16],
                                      initializer=tf.contrib.layers.xavier_initializer_conv2d(), dtype=tf.float64)
            b_conv1 = tf.Variable(tf.constant(0.05, shape=[16], dtype=tf.float64), name='conv1_biases')

            conv1 = tf.nn.conv2d(input = self.x, filter=w_conv1, padding="SAME", strides=[1, 1, 1, 1])
            conv1 = tf.nn.relu(conv1 + b_conv1)

        with tf.name_scope('Pool1'):
            pool1 = tf.nn.max_pool(value=conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        with tf.name_scope('Conv2'):
            w_conv2 = tf.get_variable("conv2_weights", shape=[5, 5, 16, 36],
                                      initializer=tf.contrib.layers.xavier_initializer_conv2d(), dtype=tf.float64)
            b_conv2 = tf.Variable(tf.constant(0.05, shape=[36], dtype=tf.float64), name='conv2_biases')

            conv2 = tf.nn.conv2d(input=pool1, filter=w_conv2, padding="SAME", strides=[1, 1, 1, 1])
            conv2 += b_conv2

        with tf.name_scope('Pool2'):
            pool2 = tf.nn.max_pool(value=conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        with tf.name_scope('FullyConnected1'):
            w_fc1 = tf.get_variable("fc1_weights", shape=[1764, 128],
                                    initializer = tf.contrib.layers.xavier_initializer_conv2d(), dtype=tf.float64)
            b_fc1 = tf.Variable(tf.constant(0.05, shape=[128], dtype=tf.float64), name='fc1_biases')

            flatten_input = tf.reshape(pool2, [-1, 1764])
            fc1 = tf.nn.relu(tf.matmul(flatten_input, w_fc1) + b_fc1)

        with tf.name_scope('FullyConnected2'):
            w_fc2 = tf.get_variable("fc2_weights", shape=[128, 10],
                                    initializer = tf.contrib.layers.xavier_initializer_conv2d(), dtype=tf.float64)
            b_fc2 = tf.Variable(tf.constant(0.05, shape=[10], dtype=tf.float64), name='fc2_biases')

            logits = tf.matmul(fc1, w_fc2) + b_fc2
            probabilities = tf.nn.softmax(logits)
            predictions   = tf.argmax(probabilities, axis=1)

        with tf.name_scope("cross_entropy"):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits,
                                                                                  labels = self.y),
                                  name="cross_entropy")
        with tf.name_scope("train"):
            self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        with tf.name_scope("accuracy"):
            y_cls = tf.argmax(self.y, axis = 1)
            correct_prediction = tf.equal(predictions, y_cls)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

    def train(self, X, Y, subpolicies = None):
        for i in range(self.MAXIT):
            x_batch, y_batch = random_batch(X, Y, self.BATCH_SIZE)
            if i % 10 == 0:
                [train_accuracy, l] = self.model_sess.run([self.accuracy, self.loss], feed_dict={self.x: x_batch, self.y: y_batch})
                print("Iteration {:>4}, Accuracy: {:>6}, Loss: {:>6}".format(i, train_accuracy, l))
            self.model_sess.run(self.train_step, feed_dict={self.x: x_batch, self.y: y_batch})

    def test(self, X, Y):
        batches = create_batches(X, Y, batch_size = self.BATCH_SIZE)
        n_batches = len(batches)
        acumulated_acc = 0
        for i, batch in enumerate(batches):
            print("Evaluating NeuralNet on batch {}/{}".format(i, n_batches))
            acumulated_acc += self.model_sess.run(self.accuracy, feed_dict = {self.x: batch[0], self.y: batch[1]})

        return acumulated_acc / n_batches

if __name__ == '__main__':
    dataset = MNIST(r'C:/Users/Eduardo Montesuma/PycharmProjects/AutoAugment/data/MNIST')

    Xtr = dataset.x_train_rgb
    Xts = dataset.x_test_rgb

    Ytr = dataset.y_train
    Yts = dataset.y_test

    mnist_model = MNISTModel()
    start = time.time()
    mnist_model.train(Xtr, Ytr)
    end = time.time()
    print("Training took: {} minutes".format(np.round((end - start)/60, 2)))
    print(mnist_model.test(Xts, Yts))
