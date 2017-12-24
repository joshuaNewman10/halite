import os
import tensorflow as tf
import numpy as np

from ml.config import PLANET_MAX_NUM, PER_PLANET_FEATURES
from ml.classifier.base import Classifier

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '99'
tf.logging.set_verbosity(tf.logging.ERROR)


class StarterNet(Classifier):
    name = 'neural_net'

    FIRST_LAYER_SIZE = 12
    SECOND_LAYER_SIZE = 6


    def __init__(self, model_dir, cached_model=None):
        super(StarterNet, self).__init__(model_dir)

        self._graph = tf.Graph()

        with self._graph.as_default():
            self._session = tf.Session()
            self._features = tf.placeholder(
                dtype=tf.float32,
                name='target_distribution',
                shape=(None, PLANET_MAX_NUM)
            )

            # target_distribution describes what the bot did in a real game.
            # For instance, if it sent 20% of the ships to the first planet and 15% of the ships to the second planet,
            # then expected_distribution = [0.2, 0.15 ...]
            self._target_distribution = tf.placeholder(
                dtype=tf.float32,
                name='target_distribution',
                shape=(None, PLANET_MAX_NUM)
            )

            # Combine all the planets from all the frames together, so it's easier to share
            # the weights and biases between them in the network.
            flattened_frames = tf.reshape(self._features, [-1, PER_PLANET_FEATURES])

            first_layer = tf.contrib.layers.fully_connected(inputs=flattened_frames, num_outputs=self.FIRST_LAYER_SIZE)
            second_layer = tf.contrib.layers.fully_connected(inputs=first_layer, num_outputs=self.SECOND_LAYER_SIZE)
            third_layer = tf.contrib.layers.fully_connected(input=second_layer, num_outputs=1,
                                                            activation_fn=None)  # linear activation

            # group planets back in frames
            logits = tf.reshape(third_layer, [-1, PLANET_MAX_NUM])

            self._prediction_normalized = tf.nn.softmax(logits)

            self._loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self._target_distribution)
            )

            self._optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self._loss)
            self._saver = tf.train.Saver()

            if cached_model:
                self._saver.restore(self._session)
            else:
                self._session.run(tf.global_variables_initializer())

    def fit(self, X, y):
        loss, optimizer = self._session.run(
            fetches=[self._loss, self._optimizer],
            feed_dict={
                self._features: self._normalize_input(X),
                self._target_distribution: y}
        )

        return loss

    def predict(self, X):
        predictions = self._session.run(
            fetches=self._prediction_normalized,
            feed_dict={
                self._features: self._normalize_input(np.array(X))
            }
        )

        return predictions[0]

    def compute_loss(self, X, y):
        return self._session.run(
            fetches=self._loss,
            feed_dict={
                self._features: self._normalize_input(X),
                self._target_distribution: y
            }
        )

    def save(self, file_name):
        file_path = os.path.join(self.model_dir, file_name)
        self._saver.save(self._session, file_path)

    def _normalize_input(self, X):
        shape = X.shape

        if len(shape) != 3:
            raise ValueError()

        if shape[1] != PLANET_MAX_NUM:
            raise ValueError()

        if shape[2] != PER_PLANET_FEATURES:
            raise ValueError()

        mean = np.expand_dims(X.mean(axis=1), axis=1)
        std = np.expand_dims(X.std(axis=1), axis=1)
        return (X - mean) / (std + 1e-6)  # avoid div 0 error
