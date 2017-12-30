from ml.classifier.starter import StarterNet
from ml.classifier.augmented_starter import AugmentedStarterNet
from ml.classifier.qlearn import QlearnNet

CLASSIFIER_REGISTRY = {
    StarterNet.name: StarterNet,
    QlearnNet.name: QlearnNet,
    AugmentedStarterNet.name: AugmentedStarterNet
}
