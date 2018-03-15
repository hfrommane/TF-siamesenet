import tensorflow as tf


class SIAMESE(object):
    def siamesenet(self, input, reuse=False):
        with tf.name_scope("model"):
            with tf.variable_scope("conv1") as scope:
                net = tf.contrib.layers.conv2d(input, 32, [7, 7], activation_fn=tf.nn.relu, padding='SAME',
                                               weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                               scope=scope, reuse=reuse)
                net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

            with tf.variable_scope("conv2") as scope:
                net = tf.contrib.layers.conv2d(net, 64, [5, 5], activation_fn=tf.nn.relu, padding='SAME',
                                               weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                               scope=scope, reuse=reuse)
                net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

            with tf.variable_scope("conv3") as scope:
                net = tf.contrib.layers.conv2d(net, 128, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
                                               weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                               scope=scope, reuse=reuse)
                net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

            with tf.variable_scope("conv4") as scope:
                net = tf.contrib.layers.conv2d(net, 256, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
                                               weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                               scope=scope, reuse=reuse)

            with tf.variable_scope("conv5") as scope:
                net = tf.contrib.layers.conv2d(net, 256, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
                                               weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                               scope=scope, reuse=reuse)
                net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
                output_0 = tf.contrib.layers.flatten(net)

            with tf.variable_scope("conv6") as scope:
                net = tf.contrib.layers.conv2d(net, 512, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
                                               weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                               scope=scope, reuse=reuse)

            with tf.variable_scope("conv7") as scope:
                net = tf.contrib.layers.conv2d(net, 512, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
                                               weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                               scope=scope, reuse=reuse)
                net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
                output_1 = tf.contrib.layers.flatten(net)

            with tf.variable_scope("conv8") as scope:
                net = tf.contrib.layers.conv2d(net, 32, [3, 3], activation_fn=None, padding='SAME',
                                               weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                               scope=scope, reuse=reuse)
                net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

            output_2 = tf.contrib.layers.flatten(net)

            net = tf.concat([output_0, output_1, output_2], 1)

            # add hidden layer1
            hidden_Weights = tf.Variable(tf.truncated_normal([11136, 2048], stddev=0.1))
            hidden_biases = tf.Variable(tf.constant(0.1, shape=[2048]))
            net = tf.nn.relu(tf.matmul(net, hidden_Weights) + hidden_biases)

            # add hidden layer2
            hidden_Weights = tf.Variable(tf.truncated_normal([2048, 128], stddev=0.1))
            hidden_biases = tf.Variable(tf.constant(0.1, shape=[128]))
            net = tf.nn.relu(tf.matmul(net, hidden_Weights) + hidden_biases)

        return net

    def contrastive_loss(self, model1, model2, y):
        with tf.name_scope("output"):
            output_difference = tf.abs(model1 - model2)
            W = tf.Variable(tf.random_normal([128, 1], stddev=0.1), name='W')
            b = tf.Variable(tf.zeros([1, 1]) + 0.1, name='b')
            y_ = tf.nn.sigmoid(tf.matmul(output_difference, W) + b, name='distance')

        # CalculateMean loss
        with tf.name_scope("loss"):
            losses = -(y * tf.log(y_) + (1 - y) * tf.log(1 - y_))
            loss = tf.reduce_mean(losses)

        return model1, model2, y_, loss
