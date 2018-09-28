# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

def create_dataset(size=1024, dimensions=2, mean=0, scale=1, separate=False):
    """
    Creates data set consisting of two input vectors and an output vector
    representing the velocity (difference) between the input vectors
    
    Vectors generated are normally distributed with a mean at the origin
    by default
    
    Args:
      size: number of examples
      dimension: 2-d points
      mean: centred at origin if 0
      scale: std
      separate: false (default) flattens the example vectors. Otherwise,
        you'll get a list of lists representing the two vectors

    Returns:
      examples, labels
    """
    
    examples = []
    labels = []
    
    for i in range(size):
        example_left = np.random.normal(size=dimensions, scale=scale, loc=mean)
        example_right = np.random.normal(size=dimensions, scale=scale, loc=mean)
        
        label = example_right - example_left # this equates to velocity
        
        if separate:
            examples.append([example_left, example_right])
        else:
            examples.append([*example_left, *example_right])
            
        labels.append(label)
        
        
    return examples, labels

def write_tfrecord(examples, labels, filename):
    """
    Writes a tfrecord to filename with the given examples, labels

    Args:
      example: an array or list of numbers (must cast to float)
      labels: an array or list of numbers (must cast to float)
      filename: output tfrecord filename
    """
    with tf.python_io.TFRecordWriter(filename) as writer:
        for example, label in zip(examples, labels):
            # Create a feature dictionary
            feature = {
                    # See https://github.com/tensorflow/tensorflow/issues/9554#issuecomment-298761938
                    "example" : tf.train.Feature(float_list=tf.train.FloatList(value=[float(x) for x in example])),
                    "label" :  tf.train.Feature(float_list=tf.train.FloatList(value=[float(x) for x in label]))
                   }

            tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
            
            writer.write(tf_example.SerializeToString())
            
def load_tfrecord(filename):
    """
    This is heavily based off the EM code

    Args:
      filename: name of tfrecord file to load

    Returns:
      x, y: example, label tuple
    """
    # Create the file queue
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    
    # Parse single example
    features = tf.parse_single_example(serialized_example,
                                   features={
                                       'example': tf.FixedLenFeature([], tf.float32),
                                       'label': tf.FixedLenFeature([], tf.float32) # May need to specifiy the length of the array ie tf.FixedLenFeature([16], tf.float32)
                                   })
        
    # Label cast
    label = features['label']
    label = tf.cast(label, tf.float32)
    
    # example cast
    example = features["example"]
    example = tf.cast(example, tf.float32)
    #pose = tf.reshape(pose, [4])
    
    x, y = tf.train.shuffle_batch([example, label], allow_smaller_final_batch=False, batch_size=128, capacity=1028, min_after_dequeue=512)#, num_threads=cfg.num_threads, batch_size=cfg.batch_size, capacity=cfg.batch_size * 64,
                              #min_after_dequeue=cfg.batch_size * 32, )
    
    return x, y
            
if __name__ == "__main__":
    # Generate some data!
    train_filename = "train.tfrecords"
    test_filename = "test.tfrecords"
    
    train_size = 8192
    test_size = 1024
    
    # Train
    examples, labels = create_dataset(train_size)
    write_tfrecord(examples, labels, train_filename)
    
    # Test
    examples, labels = create_dataset(test_size)
    write_tfrecord(examples, labels, test_filename)
    