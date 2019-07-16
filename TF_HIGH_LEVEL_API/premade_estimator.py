import argparse
import tensorflow as tf

from recommend_system_practise.TF_HIGH_LEVEL_API import iris_data


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int, help='number of training steps')

def main(argv):
    args = parser.parse_args(argv[1:])

    (train_x, train_y), (test_x, test_y) = iris_data.load_data()

    my_feature_cols = []
    for key in train_x.keys():
        my_feature_cols.append(tf.feature_column.numeric_column(key=key))

    my_checkpoint_config = tf.estimator.RunConfig(
        save_checkpoints_secs=1200,
        keep_checkpoint_max=10
    )

    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_cols,
        hidden_units=[10, 10],
        n_classes=3,
        model_dir='models/iris',
        config=my_checkpoint_config
    )

    a = dict(train_x)
    classifier.train(
        input_fn=lambda :iris_data.train_input_fn(train_x, train_y, args.batch_size),
        steps=args.train_steps
    )

    eval_result = classifier.evaluate(
        input_fn=lambda :iris_data.eval_input_fn(test_x, test_y, args.batch_size)
    )

    print(f'test set accuracy: {eval_result}\n')

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)