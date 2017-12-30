from ml.helpers.training.base import TrainingHelper
from ml.helpers.training.qlearn import QlearningTrainingHelper

TRAINING_HELPER_REGISTRY = {
    TrainingHelper.name: TrainingHelper,
    QlearningTrainingHelper.name: QlearningTrainingHelper
}
