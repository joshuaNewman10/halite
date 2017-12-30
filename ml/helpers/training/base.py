import json
import os
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=os.getenv('LOG_LEVEL', logging.DEBUG))
logger = logging.getLogger(__name__)

VALIDATION_MESSAGE = 'Step: {step_num}, cross val loss {cv_loss} training loss {t_loss}'


class TrainingHelper:
    TRAINING_DATA_SPLIT = 0.85

    def __init__(self, model, parser, model_dir, training_data_dir, model_file_name, data_dir, max_num_replays, checkpoint_step_num=5000):
        self._checkpoint_step_num = checkpoint_step_num
        self._model = model
        self._parser = parser
        self._model_dir = model_dir
        self._model_file_name = model_file_name
        self._training_data_dir = training_data_dir
        self._replay_dir = data_dir
        self._max_num_replays = max_num_replays

    def fit(self, num_steps, minibatch_size, loss_step_num):
        epoch_training_loss = []

        game_data = self.load_data()
        game_data = game_data[:self._max_num_replays]

        X, y = self._parser.parse(game_data)
        train_X, train_y, val_X, val_y = self.split_data(X, y)

        for step_num in range(num_steps):
            X, y = self._get_minibatch(step_num, minibatch_size, train_X, train_y)
            training_loss = self._model.fit(X, y)

            if step_num % loss_step_num == 0 or step_num == num_steps - 1:
                validation_loss = self._model.compute_loss(val_X, val_y)
                logger.info(VALIDATION_MESSAGE.format(step_num=step_num, cv_loss=validation_loss, t_loss=training_loss))
                epoch_training_loss.append((step_num, training_loss, validation_loss))

            if step_num % self._checkpoint_step_num == 0 or step_num == num_steps - 1:
                self.save(self._model_file_name, step_num)

        training_stats = self.get_training_stats_plot(epoch_training_loss)
        self.save_training_stats(training_stats)

    def load_data(self):
        training_data = []

        game_files = os.listdir(self._replay_dir)
        for game_file in game_files:
            file_path = os.path.join(self._replay_dir, game_file)

            if not self._is_replay_file(file_path):
                logger.info('Skipping over non replay file %s', file_path)
                continue

            with open(file_path) as f:
                game_data = f.read()
                try:
                    game_data = json.loads(game_data)
                except json.JSONDecodeError:
                    logger.warning('Failed to load file %s', file_path)
                    continue
                training_data.append(game_data)

        logger.info('Found %s replays', len(training_data))
        return training_data

    def _is_replay_file(self, file_path):
        file_name = os.path.basename(file_path)
        return os.path.isfile(file_path) and 'replay' in file_name

    def save(self, model_name, step_num):
        step_num = str(step_num)
        model_path = os.path.join(self._model_dir, model_name + '_' + step_num)
        self._model.save(model_path)

    def get_training_stats_plot(self, epoch_training_loss):
        cf = pd.DataFrame(epoch_training_loss, columns=['step', 'training_loss', 'cv_loss'])
        fig = cf.plot(x='step', y=['training_loss', 'cv_loss']).get_figure()
        return fig

    def save_training_stats(self, fig):
        curve_path = os.path.join(self._model_dir, self._model_file_name + '_training_loss_fig.png')
        fig.savefig(curve_path)

    def split_data(self, X, y):
        num_samples = len(X)
        split_integer = int(self.TRAINING_DATA_SPLIT * num_samples)

        permutation = np.random.permutation(num_samples)
        X, y = X[permutation], y[permutation]

        train_X, train_y = X[:split_integer], y[:split_integer]
        val_X, val_y = X[split_integer:], y[split_integer:]

        return train_X, train_y, val_X, val_y

    def _get_minibatch(self, step_num, minibatch_size, X, y):
        start = (step_num * minibatch_size) % len(X)
        end = start + minibatch_size
        return X[start:end], y[start:end]
