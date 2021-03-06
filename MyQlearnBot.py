from ml.agent.registry import AGENT_REGISTRY
from ml.classifier.registry import CLASSIFIER_REGISTRY
from ml.runner.base import GameRunner


AGENT_NAME = 'qlearn'
MODEL_FILE = '/Users/jnewman/Projects/Banjo/halite/scripts/models/josh_qlearn_bot_ckpt_99999'
CLASSIFIER_NAME = 'qlearn'


def main(agent_name, classifier_name, model_file):
    classifier = CLASSIFIER_REGISTRY[classifier_name](model_file=model_file)
    agent = AGENT_REGISTRY[agent_name](classifier)
    runner = GameRunner(agent)
    runner.run()


if __name__ == '__main__':
    main(
        agent_name=AGENT_NAME,
        classifier_name=CLASSIFIER_NAME,
        model_file=MODEL_FILE
    )
