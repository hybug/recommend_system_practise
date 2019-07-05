import pandas as pd
import tensorflow as tf

train_url = 'http://download.tensorflow.org/data/iris_training.csv'
test_url = 'http://download.tensorflow.org/data/iris_test.csv'

csv_column_names = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
species = ['Setosa', 'Versicolor', 'Virginica']

def maybe_download():
    train_path = tf.keras.utils.get_file(train_url.split('/')[-1], train_url)
    test_path = tf.keras.utils.get_file(test_url.split('/')[-1], test_url)
    return train_path, test_path

def load_data(y_name='Species'):
    train_path, test_path = maybe_download()

    train = pd.read_csv(train_path, names=csv_column_names, header=0)
    train_x, train_y = train, train.pop(y_name)

    test = pd.read_csv(test_path, names=csv_column_names, header=0)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)

def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset

def eval_input_fn(features, labels, batch_size):
    features = dict(features)
    if labels is  None:
        inputs = features
    else:
        inputs = (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    assert batch_size is not None
    dataset = dataset.batch(batch_size)

    return dataset

csv_types = [[0.0], [0.0], [0.0], [0]]

def _parse_line(line):
    fields = tf.decode_csv(line, record_defaults=csv_types)
    features = dict(zip(csv_column_names, fields))
    label = features.pop('Species')
    return features, label

def csv_input_fn(csv_path, batch_size):
    dataset = tf.data.TextLineDataset(csv_path).skip(1)
    dataset = dataset.map(_parse_line)
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset