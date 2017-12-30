import logging
import os

from argparse import ArgumentParser

from ml.agent.registry import AGENT_REGISTRY
from ml.classifier.registry import CLASSIFIER_REGISTRY
from ml.runner.base import GameRunner

logging.basicConfig(level=os.getenv('LOG_LEVEL', logging.DEBUG))
logging.getLogger(__name__)


def main(agent_name, classifier_name, training_data_dir, model_dir, cached_model_file):
    classifier = CLASSIFIER_REGISTRY[classifier_name](model_dir, cached_model_file)
    agent = AGENT_REGISTRY[agent_name](classifier)
    runner = GameRunner(agent)
    runner.run()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--agent_name', required=True)
    parser.add_argument('--cached_model_file', required=True)
    parser.add_argument('--classifier_name', required=True)
    parser.add_argument('--training_data_dir', required=False, default=os.path.join(os.getcwd(), 'training_data_dir'))
    parser.add_argument('--model_dir', required=False, default=os.path.join(os.getcwd(), 'models'))

    args = parser.parse_args()
    main(args.agent_name, args.classifier_name, args.training_data_dir, args.model_dir, args.cached_model_file)
