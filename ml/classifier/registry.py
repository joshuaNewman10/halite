from ml.classifier.starter import Starter
from ml.classifier.augmented_starter import AugmentedStarter
from ml.classifier.qlearn import QlearnNet

CLASSIFIER_REGISTRY = {
    Starter.name: Starter,
    QlearnNet.name: QlearnNet,
    AugmentedStarter.name: AugmentedStarter
}
