import os

from argparse import ArgumentParser

from ml.parser.registry import PARSER_REGISTRY
from ml.classifier.registry import CLASSIFIER_REGISTRY
from ml.helpers.training.registry import TRAINING_HELPER_REGISTRY


def main(classifier_name, training_helper_name, parser_name, replay_dir, training_data_dir, model_dir, model_file_name,
         max_num_replays, minibatch_size, num_steps, loss_step_num):

    classifier = CLASSIFIER_REGISTRY[classifier_name](model_dir)
    parser = PARSER_REGISTRY[parser_name](training_data_dir)

    training_helper = TRAINING_HELPER_REGISTRY[training_helper_name](
        data_dir=replay_dir,
        training_data_dir=training_data_dir,
        parser=parser,
        model=classifier,
        max_num_replays=max_num_replays,
        model_dir=model_dir,
        model_file_name=model_file_name

    )

    training_helper.fit(num_steps=num_steps, minibatch_size=minibatch_size, loss_step_num=loss_step_num)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--classifier_name', required=True)
    parser.add_argument('--parser_name', required=True)
    parser.add_argument('--training_helper_name', required=True)
    parser.add_argument('--model_file_name', required=True)
    parser.add_argument('--max_num_replays', required=False, type=int, default=None)
    parser.add_argument('--minibatch_size', required=False, type=int, default=64)
    parser.add_argument('--loss_step_num', required=False, type=int, default=24)
    parser.add_argument('--num_steps', required=False, type=int, default=64)
    parser.add_argument('--replay_dir', required=False, default=os.path.join(os.getcwd(), 'replays'))
    parser.add_argument('--training_data_dir', required=False, default=os.path.join(os.getcwd(), 'training_data/data'))
    parser.add_argument('--model_dir', required=False, default=os.path.join(os.getcwd(), 'models'))

    args = parser.parse_args()
    main(
        classifier_name=args.classifier_name,
        parser_name=args.parser_name,
        training_helper_name=args.training_helper_name,
        replay_dir=args.replay_dir,
        training_data_dir=args.training_data_dir,
        model_dir=args.model_dir,
        model_file_name=args.model_file_name,
        max_num_replays=args.max_num_replays,
        minibatch_size=args.minibatch_size,
        num_steps=args.num_steps,
        loss_step_num=args.loss_step_num
    )
