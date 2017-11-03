# kernel method


import tensorflow as tf
import numpy as np
import time

FLY_TRAINING = "train.csv"
FLY_TEST = "test.csv"

def get_input_fn(dataset_split, batch_size, capacity=10000, min_after_dequeue=3000):

  def _input_fn():
    images_batch, labels_batch = tf.train.shuffle_batch(
        tensors=[dataset_split.data.astype(np.float32), dataset_split.target.astype(np.int32)],
        batch_size=batch_size,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue,
        enqueue_many=True,
        num_threads=2)
    features_map = {'images': images_batch}
    return features_map, labels_batch

  return _input_fn


training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=FLY_TRAINING,
      target_dtype=np.int,
      features_dtype=np.int)

test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=FLY_TEST,
      target_dtype=np.int,
      features_dtype=np.int)


train_input_fn = get_input_fn(training_set, batch_size=2560)
eval_input_fn = get_input_fn(test_set, batch_size=5000)


# Specify the feature(s) to be used by the estimator. This is identical to the
# code used for the LinearClassifier.
image_column = tf.contrib.layers.real_valued_column('images', dimension=12)
optimizer = tf.train.FtrlOptimizer(
   learning_rate=50.0, l2_regularization_strength=0.001)

kernel_mapper = tf.contrib.kernel_methods.RandomFourierFeatureMapper(
  input_dim=12, output_dim=2000, stddev=25.0, name='rffm')
kernel_mappers = {image_column: [kernel_mapper]}
estimator = tf.contrib.kernel_methods.KernelLinearClassifier(
   n_classes=22, optimizer=optimizer, kernel_mappers=kernel_mappers)

# Train.
start = time.time()
estimator.fit(input_fn=train_input_fn, steps=100000)
end = time.time()
print('Elapsed time: {} seconds'.format(end - start))

# Evaluate and report metrics.
eval_metrics = estimator.evaluate(input_fn=eval_input_fn, steps=1)
print(eval_metrics)
