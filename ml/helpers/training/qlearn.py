import logging
import numpy as np
import os
import random

from itertools import islice

from ml.helpers.training.base import TrainingHelper, VALIDATION_MESSAGE

logging.basicConfig(level=os.getenv('LOG_LEVEL', logging.DEBUG))
logger = logging.getLogger(__name__)


class QlearningTrainingHelper(TrainingHelper):
    name = 'qlearn'

    gamma = 0.2
    batch_size = 64

    def fit(self, num_steps, minibatch_size, loss_step_num):
        X = []
        y = []
        epoch_training_loss = []

        game_data = self.load_data()
        game_data = islice(game_data, 0, self._max_num_replays)
        env_history = self._parser.parse(game_data)

        env_state_mini_batch = random.sample(env_history, k=self.batch_size)

        for env_step in env_state_mini_batch:
            observation = env_step['observation']
            observation = self.format_observation(observation)

            next_observation = env_step['next_observation']
            next_observation = self.format_observation(next_observation)

            reward = env_step['reward']
            allocations = env_step['allocations']
            allocations = self.format_target(allocations)
            target = (reward + self.gamma * np.amax(self._model.predict(next_observation)[0]))

            future_discounted_reward = self._predict_future_discounted_reward(observation, target, allocations)
            X.append(observation)
            y.append(future_discounted_reward)

        X = np.array(X)
        y = np.array(y)

        train_X, train_y, val_X, val_y = self.split_data(X, y)
        for step_num in range(num_steps):
            X, y = self._get_minibatch(step_num, minibatch_size, X, y)
            training_loss = self._model.fit(X, y)

            if step_num % loss_step_num == 0 or step_num == num_steps - 1:
                validation_loss = self._model.compute_loss(val_X, val_y)
                logger.info(VALIDATION_MESSAGE.format(step_num=step_num, cv_loss=validation_loss, t_loss=training_loss))
                epoch_training_loss.append((step_num, training_loss, validation_loss))

            if step_num % self._checkpoint_step_num == 0 or step_num == num_steps - 1:
                self.save(self._model_file_name, step_num)

        training_stats = self.get_training_stats_plot(epoch_training_loss)
        self.save_training_stats(training_stats)

    def _predict_future_discounted_reward(self, observation, target, allocations):
        highest_allocation_ix = np.argmax(allocations, axis=0)
        target_future_discounted_rewards = self._model.predict(X=observation)

        target_future_discounted_rewards[highest_allocation_ix] = target
        return target_future_discounted_rewards
