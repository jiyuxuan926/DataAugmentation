import time
import operations
import numpy as np
import tensorflow as tf
import PIL.Image as Image
import keras.layers as layers
import keras.models as models
import keras.initializers as initializers
import keras.backend as backend

from data.mnist import MNIST
from models.Mnist_Model import MNISTModel

tf_session = tf.Session()
backend.set_session(tf_session)

dataset = MNIST(r'C:/Users/Eduardo Montesuma/PycharmProjects/AutoAugment/data/MNIST')
reduced_num_train = dataset.num_train // 10
rand_samples = np.random.randint(low = 0, high = dataset.num_train, size = reduced_num_train)

Xtr = dataset.x_train_rgb
Ytr = dataset.y_train

Xtr = Xtr[rand_samples]
Ytr = Ytr[rand_samples]

Xts = dataset.x_test_rgb
Yts = dataset.y_test

transformations = operations.get_transformations(Xtr)

N_SUBPOL = 5
N_UNITS  = 126
N_OPS    = 2
N_TYPES  = 16
N_PROBS  = 11
N_MAG    = 10

DATA_BATCH_SIZE  = 128
NUM_DATA_BATCHES = len(Xtr) // DATA_BATCH_SIZE

MODEL_EPOCHS = 500
CONTROLLER_EPOCHS = 500

class Operation:
    def __init__(self, type, probability, magnitude):
        self.type = type
        t = transformations[self.type]
        self.prob = probability / (N_PROBS - 1)
        m = magnitude / (N_MAG - 1)
        self.magn = m * (t[2] - t[1]) + t[1]
        self.transformation = t[0]

    def __call__(self, image_set):
        transformed_set = []
        for image in image_set:
            if np.random.rand() < self.prob:
                image = Image.fromarray(image)
                image = self.transformation(image, self.magn)
            transformed_set.append(np.array(image))
        return np.array(transformed_set)

    def __str__(self):
        return "Operation: {} | Probability: {} | Magnitude: {}".format(self.type, self.prob, self.magn)

class Subpolicy:
    def __init__(self, *ops):
        self.operations = ops

    def __call__(self, image_set):
        for op in self.operations:
            image_set = op(image_set)
        return image_set

class Controller:
    def __init__(self):
        self.model = self.controller_model()
        self.scale = tf.placeholder(tf.float32, ())
        self.grads = tf.gradients(self.model.outputs, self.model.trainable_weights)
        self.grads = [- self.scale * g for g in self.grads]
        self.grads = zip(self.grads, self.model.trainable_weights)
        self.optmizer = tf.train.AdamOptimizer().apply_gradients(self.grads)

    def controller_model(self):
        input_layer = layers.Input(shape = (N_SUBPOL, 1))
        initializer = initializers.glorot_normal(seed = None)
        rec_layer = layers.LSTM(N_UNITS, recurrent_initializer = initializer, return_sequences = True,
                                     name = 'Controller_Network')(input_layer)

        outputs = []
        for i in range(N_OPS):
            name = "operation_" + str(i+1)
            outputs.extend([
                layers.Dense(N_TYPES, activation ='softmax', name =name + '_type')(rec_layer),
                layers.Dense(N_PROBS, activation ='softmax', name =name + '_prob')(rec_layer),
                layers.Dense(N_MAG, activation ='softmax', name =name + '_magn')(rec_layer)
            ])

        return models.Model(input_layer, outputs)

    def train(self, mem_softmax, mem_rewards):
        session = backend.get_session()
        min_reward = np.min(mem_rewards)
        max_reward = np.max(mem_rewards)
        initial_input = np.zeros((1, N_SUBPOL, 1))
        dict_inputs = {self.model.input : initial_input}
        for distributions, r in zip(mem_softmax, mem_rewards):
            scale = (r - max_reward) / (max_reward - min_reward)
            dict_outputs = {_output: s for _output, s in zip(self.model.outputs, distributions)}
            dict_scales  = {self.scale : scale}
            session.run(self.optmizer, feed_dict = {**dict_outputs, **dict_scales, **dict_inputs})
        return self

    def predict(self, size):
        initial_input = np.zeros((1, size, 1))
        distributions = self.model.predict(initial_input)
        subpolicies = []
        for i in range(size):
            operations = []
            for j in range(N_OPS):
                op = distributions[3 * j : 3 * (j + 1)]
                op = [o[0, i, :].argmax() for o in op]
                operations.append(Operation(*op))
            subpolicies.append(Subpolicy(*operations))
        return distributions, subpolicies

def dataset_generator(subpolicies, X, Y):
    while True:
        ix = np.arange(len(X))
        np.random.shuffle(ix)
        for i in range(NUM_DATA_BATCHES):
            _ix = ix[i * DATA_BATCH_SIZE : (i + 1) * DATA_BATCH_SIZE]
            _X  = X[_ix]
            _Y  = Y[_ix]
            subpolicy = np.random.choice(subpolicies)
            _X = subpolicy(_X)
            _X = _X.astype(np.float32) / 255
            yield _X, _Y


mem_softmaxes = []
mem_accuracies = []

controller = Controller()

for epoch in range(CONTROLLER_EPOCHS):
    print("Controller Epoch: {}/{}".format(epoch + 1, CONTROLLER_EPOCHS))

    softmaxes, subpolicies = controller.predict(N_SUBPOL)
    for i, subpolicy in enumerate(subpolicies):
        print("Subpolicy: \t{}".format(i+1))
        print(subpolicy)
    mem_softmaxes.append(softmaxes)

    child = MNISTModel(MAXIT = MODEL_EPOCHS, BATCH_SIZE = 128)
    tic = time.time()
    child.train(Xtr, Ytr, subpolicies)
    toc = time.time()

    accuracy = child.test(Xts, Yts)
    print('\tTrain Time\t Train Accuracy')
    print('\t{}\t{}'.format(time, accuracy))
    mem_accuracies.append(accuracy)

    if len(mem_softmaxes) > 5:
        controller.train(mem_softmaxes, mem_accuracies)


