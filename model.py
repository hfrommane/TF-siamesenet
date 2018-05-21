import tensorflow as tf

variables_dict = {
    "hidden_Weights": tf.Variable(tf.truncated_normal([144, 128], stddev=0.1), name="hidden_Weights"),
    "hidden_biases": tf.Variable(tf.constant(0.1, shape=[128]), name="hidden_biases")
}


class SIAMESE(object):
    def siamesenet(self, input, reuse=False):
        with tf.name_scope("model"):
            with tf.variable_scope("conv1") as scope:
                conv1 = tf.layers.conv2d(input, filters=64, kernel_size=[5, 5], strides=[1, 1],
                                         padding='SAME', activation=tf.nn.relu, reuse=reuse, name=scope.name)
                pool1 = tf.layers.max_pooling2d(conv1, pool_size=[3, 3], strides=[2, 2],
                                                padding='SAME', name='pool1')

            with tf.variable_scope("conv2") as scope:
                conv2 = tf.layers.conv2d(pool1, filters=128, kernel_size=[5, 5], strides=[1, 1],
                                         padding='SAME', activation=tf.nn.relu, reuse=reuse, name=scope.name)
                pool2 = tf.layers.max_pooling2d(conv2, pool_size=[3, 3], strides=[2, 2],
                                                padding='SAME', name='pool2')

            with tf.variable_scope("conv3") as scope:
                conv3 = tf.layers.conv2d(pool2, filters=256, kernel_size=[3, 3], strides=[1, 1],
                                         padding='SAME', activation=tf.nn.relu, reuse=reuse, name=scope.name)
                pool3 = tf.layers.max_pooling2d(conv3, pool_size=[3, 3], strides=[2, 2],
                                                padding='SAME', name='pool3')

            with tf.variable_scope("conv4") as scope:
                conv4 = tf.layers.conv2d(pool3, filters=512, kernel_size=[3, 3], strides=[1, 1],
                                         padding='SAME', activation=tf.nn.relu, reuse=reuse, name=scope.name)
                pool4 = tf.layers.max_pooling2d(conv4, pool_size=[3, 3], strides=[2, 2],
                                                padding='SAME', name='pool4')

            with tf.variable_scope("conv5") as scope:
                conv5 = tf.layers.conv2d(pool4, filters=16, kernel_size=[3, 3], strides=[1, 1],
                                         padding='SAME', activation=tf.nn.relu, reuse=reuse, name=scope.name)
                pool5 = tf.layers.max_pooling2d(conv5, pool_size=[3, 3], strides=[2, 2],
                                                padding='SAME', name='pool5')

            flattened = tf.contrib.layers.flatten(pool5)

            with tf.variable_scope("local") as scope:
                output = tf.nn.relu(tf.matmul(flattened, variables_dict["hidden_Weights"]) +
                                    variables_dict["hidden_biases"], name=scope.name)

        return output

    def contrastive_loss(self, model1, model2, y):
        with tf.name_scope("output"):
            output_difference = tf.abs(model1 - model2)
            W = tf.Variable(tf.random_normal([128, 1], stddev=0.1), name='W')
            b = tf.Variable(tf.zeros([1, 1]) + 0.1, name='b')
            y_ = tf.add(tf.matmul(output_difference, W), b, name='distance')

        # CalculateMean loss
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_)
            loss = tf.reduce_sum(losses)

        return model1, model2, y_, loss
