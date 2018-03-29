import tensorflow as tf
from dataset import *

file_positive = open('positive_pairs_path.txt', 'r')
file_negative = open('negative_pairs_path.txt', 'r')

images_positive = [line.strip() for line in file_positive.readlines()]
images_negative = [line.strip() for line in file_negative.readlines()]

images = np.asarray(images_positive + images_negative)
labels = np.append(np.ones([3000]), np.zeros([3000]))

np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(labels)))
images_shuffled = images[shuffle_indices]
labels_shuffled = labels[shuffle_indices]

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        saver = tf.train.import_meta_graph('checkpoint/model_100000.ckpt.meta')
        saver.restore(sess, 'checkpoint/model_100000.ckpt')

        left = graph.get_operation_by_name("in/left").outputs[0]
        right = graph.get_operation_by_name("in/right").outputs[0]

        distance = graph.get_operation_by_name("output/distance").outputs[0]

        image_test = []
        label_test = []
        index = 1

        # Generate batches for one epoch
        for image, label in zip(images_shuffled, labels_shuffled):
            index += 1
            image_test.append(image)
            label_test.append(label)
            if index % 100 == 0 and index > 0:
                left_test = []
                right_test = []
                for image_one in image_test:
                    line_one_list = str(image_one).split(' ')
                    left_test.append(line_one_list[0])
                    right_test.append(line_one_list[1])
                left_test_arr, right_test_arr, _ = get_batch_image_array(left_test, right_test, [])

                output_distance = sess.run([distance], feed_dict={left: left_test_arr, right: right_test_arr})
                output_distance = output_distance[0]

                true_num = 0
                for distance_one, label_one in zip(output_distance, label_test):
                    if float(distance_one) < 0.5:
                        same_flag = 0
                    else:
                        same_flag = 1
                    if label_one == same_flag:
                        true_num += 1

                print(true_num / 100.)

                image_test = []
                label_test = []
