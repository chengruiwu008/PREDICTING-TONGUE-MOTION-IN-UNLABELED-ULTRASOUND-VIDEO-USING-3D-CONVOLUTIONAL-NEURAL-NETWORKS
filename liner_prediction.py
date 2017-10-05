import numpy as np
import cv2
import tensorflow as tf

lenth = 96
pixel = lenth * lenth

def load_train_batch():
    ran = np.random.randint(0, 9925, size=10, dtype='int')
    image_out = []
    label = []
    n_pic = ran  # np.random.randint(0,5980)
    # print(n_pic)
    for i in range(10):
        image = []
        for j in range(8):
            frame_0 = cv2.imread('./syf_friendship_20170731_153206_us/%d.jpg' % (n_pic[i] + j), 0)
            # frame_0 = add_noise(frame_0, n = noise)
            frame_0 = cv2.resize(frame_0, (lenth, lenth))
            frame_0 = np.array(frame_0).reshape(-1)
            frame_0 = frame_0
            image.append(frame_0)
        # print('shape(image)', np.shape(image))
        image_out.append(image)
        # print('shape(image_out)', np.shape(image_out))
    for i in range(10):
        frame_1 = cv2.imread('./syf_friendship_20170731_153206_us/%d.jpg' % (n_pic[i] + 8), 0)
        frame_1 = cv2.resize(frame_1, (lenth, lenth))
        frame_1 = np.array(frame_1).reshape(-1)
        frame_1 = frame_1
        # frame_1 = gray2binary(frame_1)
        label.append(frame_1)
    return np.array(image_out, dtype='float'), np.array(label, dtype='float')

def load_test_batch():
    ran = np.random.randint(0, 4842, size=10, dtype='int')
    image_out = []
    label = []
    n_pic = ran
    # print(n_pic)
    for i in range(10):
        image = []
        for j in range(8):
            frame_0 = cv2.imread('./syf_dream_20170731_153920_us/%d.jpg' % (n_pic[i] + j), 0)
            # frame_0 = add_noise(frame_0, n = noise)
            frame_0 = cv2.resize(frame_0, (lenth, lenth))
            frame_0 = np.array(frame_0).reshape(-1)
            frame_0 = frame_0
            image.append(frame_0)
            # print('shape(image)', np.shape(image))
        image_out.append(image)
        # print('shape(image_out)', np.shape(image_out))
    for i in range(10):
        frame_1 = cv2.imread('./syf_dream_20170731_153920_us/%d.jpg' % (n_pic[i] + 8), 0)
        frame_1 = cv2.resize(frame_1, (lenth, lenth))
        frame_1 = np.array(frame_1).reshape(-1)
        frame_1 = frame_1
        # frame_1 = gray2binary(frame_1)
        label.append(frame_1)
    return np.array(image_out, dtype='float'), np.array(label, dtype='float'), n_pic

with tf.name_scope('input'):
    x_in = tf.placeholder(dtype='float', shape=[None,pixel,8])
with tf.name_scope('target'):
    y_out = tf.placeholder(dtype='float', shape=[None,pixel])
with tf.name_scope('liner_prediction'):
    x_in_ = tf.reshape(x_in,[-1,8])
    # print(x_in_.shape)
    w_in = tf.Variable(tf.constant(0.125, shape=[8,1]),name='W')
    # print(w_in.shape)
    result = tf.matmul(x_in_, w_in)
    # print(result.shape)
    result = tf.reshape(result,[-1,pixel,1])
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.square(tf.reshape(y_out ,[-1]) - tf.reshape(result ,[-1])))
with tf.name_scope('training'):
    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

all_saver = tf.train.Saver()
# saver = tf.train.import_meta_graph('./liner_prediction/save/data.chkp.meta')

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    writer = tf.summary.FileWriter('./liner_prediction/logs/', sess.graph)
    # saver.restore(sess, tf.train.latest_checkpoint('./liner_prediction/save/'))
    sess.run(init)
    for i in range(5000):
        batch,target = load_train_batch()
        batch = np.transpose(batch,[0,2,1])
        sess.run(optimizer, feed_dict={x_in:batch, y_out:target})
        if i % 10==0:
            cost = sess.run(loss, feed_dict={x_in:batch, y_out:target})
            print('Batch {}'.format(i),'training loss: {:.4f} '.format(cost))
            print('weight ', sess.run(w_in, feed_dict={x_in:batch, y_out:target}))
            all_saver.save(sess, './liner_prediction/save/data.chkp')
    print('optimization finished!')

    for i in range(10000):
        batch_x, batch_y, n_pic = load_test_batch()
        batch_x = np.transpose(batch_x,[0,2,1])
        img_p = sess.run(result, feed_dict={x_in: batch_x, y_out: batch_y})
        # print(img_p.shape)
        for n in range(10):
            img = np.array(img_p[n],dtype='int32').reshape([lenth,lenth])
            # print(img)
            cv2.imwrite('./liner_prediction/output_images/%d.jpg'% (n_pic[n]+8), img)
        if i % 50 == 0:
            print('%d finished' % i )
    print('all finished')

