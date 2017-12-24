from abc import abstractmethod


class Classifier:
    def __init__(self, model_dir):
        self.model_dir = model_dir

    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError()

    @abstractmethod
    def save(self, file_name):
        raise NotImplementedError()
