import os
import tensorflow as tf

from ml.config import PLANET_MAX_NUM, PER_PLANET_FEATURES
from ml.classifier.starter import StarterNet

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '99'
tf.logging.set_verbosity(tf.logging.ERROR)


class AugmentedStarterNet(StarterNet):
    name = 'augmented_starter'

    FIRST_LAYER_SIZE = 24
    SECOND_LAYER_SIZE = 12
    THIRD_LAYER_SIZE = 6

    def __init__(self, model_dir=None, model_file=None):
        super(AugmentedStarterNet, self).__init__(model_dir)

        self._graph = tf.Graph()

        with self._graph.as_default():
            self._session = tf.Session()

            self._features = tf.placeholder(
                dtype=tf.float32,
                name='input_features',
                shape=(None, PLANET_MAX_NUM, PER_PLANET_FEATURES)
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
            third_layer = tf.contrib.layers.fully_connected(inputs=second_layer, num_outputs=self.THIRD_LAYER_SIZE)
            fourth_layer = tf.contrib.layers.fully_connected(inputs=third_layer, num_outputs=1,
                                                             activation_fn=None)  # linear activation

            # group planets back in frames
            logits = tf.reshape(fourth_layer, [-1, PLANET_MAX_NUM])

            self._prediction_normalized = tf.nn.softmax(logits)

            self._loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self._target_distribution)
            )

            self._optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self._loss)
            self._saver = tf.train.Saver()

            if model_file:
                self._saver.restore(self._session, model_file)
            else:
                self._session.run(tf.global_variables_initializer())
