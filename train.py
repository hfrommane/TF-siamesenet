import tensorflow as tf
from model import SIAMESE
from dataset import *
import logging

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
                    datefmt='%b %d %H:%M')

with tf.name_scope("in"):
    left = tf.placeholder(tf.float32, [None, 72, 72, 3], name='left')
    right = tf.placeholder(tf.float32, [None, 72, 72, 3], name='right')
with tf.name_scope("similarity"):
    label = tf.placeholder(tf.int32, [None, 1], name='label')  # 1 if same, 0 if different
    label = tf.to_float(label)

left_output = SIAMESE().siamesenet(left, reuse=False)
print(left_output.shape)

right_output = SIAMESE().siamesenet(right, reuse=True)

model1, model2, distance, loss = SIAMESE().contrastive_loss(left_output, right_output, label)

global_step = tf.Variable(0, trainable=False)

train_step = tf.train.AdamOptimizer(0.0001).minimize(loss, global_step=global_step)

# saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)
    # saver.restore(sess, 'checkpoint_trained/model_3.ckpt')

    # setup tensorboard
    tf.summary.scalar('step', global_step)
    tf.summary.scalar('loss', loss)
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('train.log', sess.graph)

    left_dev_arr, right_dev_arr, similar_dev_arr = get_batch_image_array(left_dev, right_dev, similar_dev)

    # train iter
    idx = 0
    for i in range(FLAGS.train_iter):
        batch_left, batch_right, batch_similar, idx = get_batch_image_path(left_train, right_train, similar_train, idx)
        batch_left_arr, batch_right_arr, batch_similar_arr = \
            get_batch_image_array(batch_left, batch_right, batch_similar)

        _, l, summary_str = sess.run([train_step, loss, merged],
                                     feed_dict={left: batch_left_arr, right: batch_right_arr, label: batch_similar_arr})
        writer.add_summary(summary_str, i)
        logging.info("\r#%d - Loss" % i, l)

        if (i + 1) % FLAGS.validation_step == 0:
            val_distance = sess.run([distance],
                                    feed_dict={left: left_dev_arr, right: right_dev_arr, label: similar_dev_arr})
            logging.info(np.average(val_distance))

        if i % FLAGS.step == 0 and i != 0:
            saver.save(sess, "checkpoint/model_%d.ckpt" % i)
