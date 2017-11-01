from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib

import numpy as np
import tensorflow as tf

# Data sets
FLY_TRAINING = "train.csv"
FLY_TEST = "test.csv"

def main():

  # Load datasets.
  training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=FLY_TRAINING,
      target_dtype=np.int,
      features_dtype=np.int)
  test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=FLY_TEST,
      target_dtype=np.int,
      features_dtype=np.int)

  # Specify that all features have real-value data
  feature_columns = [tf.feature_column.numeric_column("x", shape=[12])]

  # Build 3 layer DNN with 10, 20, 10 units respectively.
  classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                          hidden_units=[120,240,120],
                                          n_classes=22,
                                          model_dir="/tmp/fly_model")
  # Define the training inputs
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(training_set.data)},
      y=np.array(training_set.target),
      num_epochs=None,
      shuffle=True)

  # Train model.
  classifier.train(input_fn=train_input_fn, steps=1000000)

  # Define the test inputs
  test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(test_set.data)},
      y=np.array(test_set.target),
      num_epochs=1,
      shuffle=True)

  # Evaluate accuracy.
  results = classifier.evaluate(input_fn=test_input_fn)

  #print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

  for key in sorted(results):
      print("%s: %s" % (key, results[key]))
 

if __name__ == "__main__":
    main()