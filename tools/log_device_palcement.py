# import tensorflow as tf
# # Creates a graph.
# a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
# b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
# c = tf.matmul(a, b)
# # Creates a session with log_device_placement set to True.
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# # Runs the op.
# print(sess.run(c))

# from keras import backend as K
# K.tensorflow_backend._get_available_gpus()
#
# import keras
#
# import tensorflow as tf
#
# config = tf.ConfigProto( device_count = {'GPU': 1} )
# sess = tf.Session(config=config)
# keras.backend.set_session(sess)

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

