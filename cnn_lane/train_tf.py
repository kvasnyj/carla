import tensorflow as tf
from tensorflow.contrib.layers import flatten

import numpy as np
import glob
from sklearn.model_selection import train_test_split
import cv2
from random import randint
import math

EPOCHS = 20
BATCH_SIZE = 100
rate = 0.001

def blackbox(img):
    h,w = img.shape[0], img.shape[1]
    size=0.1
    for i in range(10):
        x = randint(0, int(h*(1-size)))
        y = randint(0, int(w*(1-size)))
        img[x:x+int(size*h), y:y+int(size*w)]=0
    return img

def image_pipeline(file):
    img = cv2.imread(file)
    img = img[:390, :, :]
    h,w,c = img.shape
    img = cv2.resize(img, (int(h/5), int(w/5)))
    #img = cv2.resize(img, (64, 64)) 
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    
    data = np.asarray(img, dtype="float")
    data = data / 255. # normalization

    data = data[:,:, np.newaxis]
    return data

def get_data():
    X = []
    y = []

    txt = np.loadtxt("/home/kvasnyj/Dropbox/carla/cnn_lane/data/data.txt", delimiter=";")
    #txt = txt[txt[:, 0]<1000]

    # normalization
    l0min =  np.min(np.abs(txt[:, 1]))
    l1min =  np.min(np.abs(txt[:, 2]))
    l2min =  np.min(np.abs(txt[:, 3]))
    l0max =  np.max(np.abs(txt[:, 1]))
    l1max =  np.max(np.abs(txt[:, 2]))
    l2max =  np.max(np.abs(txt[:, 3]))
    l0range = l0max - l0min
    l1range = l1max - l1min
    l2range = l2max - l2min

    r0min =  np.min(np.abs(txt[:, 4]))
    r1min =  np.min(np.abs(txt[:, 5]))
    r2min =  np.min(np.abs(txt[:, 6]))
    r0max =  np.max(np.abs(txt[:, 4]))
    r1max =  np.max(np.abs(txt[:, 5]))
    r2max =  np.max(np.abs(txt[:, 6]))
    r0range = r0max - r0min
    r1range = r1max - r1min
    r2range = r2max - r2min

    range = [l0range, l1range, l2range, r0range, r1range, r2range]
    min = [l0min, l1min, l2min, r0min, r1min, r2min]

    for t in txt:
        file = "/home/kvasnyj/Dropbox/carla/cnn_lane/data/image_{:0>5d}.png".format(int(t[0]))
        X.append(image_pipeline(file))
        y.append((t[1:] - min) / range)

        
    print ("data loaded")
    return train_test_split(np.array(X), np.array(y), test_size=0.2)

def split2batches(batch_size, features, labels):
    assert len(features) == len(labels)
    outout_batches = []

    sample_size = len(features)
    for start_i in range(0, sample_size, batch_size):
        end_i = start_i + batch_size

        batch = [features[start_i:end_i], labels[start_i:end_i]]
        outout_batches.append(batch)

    return outout_batches

def get_model_nvidia(x): #source of the model: http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    conv1 = tf.layers.conv2d(inputs=x, filters=3, kernel_size=[5, 5], padding="same")
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(inputs=pool1, filters=24, kernel_size=[5, 5], padding="same")
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    conv3 = tf.layers.conv2d(inputs=pool2, filters=36, kernel_size=[5, 5], padding="same")
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    conv4 = tf.layers.conv2d(inputs=pool3, filters=48, kernel_size=[5, 5], padding="same")
    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

    conv5 = tf.layers.conv2d(inputs=pool4, filters=64, kernel_size=[5, 5], padding="same")
    pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)

    conv6 = tf.layers.conv2d(inputs=pool5, filters=64, kernel_size=[5, 5], padding="same")
    pool6 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2], strides=2)

    fc0 = flatten(pool6)
    fc0d = tf.layers.dropout(inputs=fc0, rate=0.2)

    fc1 = tf.layers.dense(inputs=fc0d, units=1164)
    fc1d = tf.layers.dropout(inputs=fc1, rate=0.2)

    fc2 = tf.layers.dense(inputs=fc1d, units=100)
    fc2d = tf.layers.dropout(inputs=fc2, rate=0.2)

    fc3 = tf.layers.dense(inputs=fc2d, units=50)
    fc3d = tf.layers.dropout(inputs=fc3, rate=0.2)

    fc4 = tf.layers.dense(inputs=fc3d, units=10, activation=tf.nn.elu)
    fc4d = tf.layers.dropout(inputs=fc4, rate=0.5)

    output = tf.layers.dense(inputs=fc4d, units=6, name = "output")

    return output

def eval_data(X_valid, y_valid):
    num_examples = len(X_valid)
    summary, loss = sess.run([merged, loss_op], feed_dict={x: X_valid, y: y_valid})

    return summary, loss

X_train, X_test, y_train, y_test =  get_data()
n_train = len(X_train)
n_test = len(X_test)
n = n_train + n_test

print("size: %s, train: %s, test: %s" % (n, n_train, n_test))

image_shape = X_train[0].shape
x = tf.placeholder(tf.float32, (None, image_shape[0], image_shape[1], image_shape[2]), name="X")
y = tf.placeholder(tf.float32, (None, 6))
learning_rate = tf.placeholder(tf.float32, shape=[])

model = get_model_nvidia(x)

mse_op = tf.losses.mean_squared_error(predictions = model, labels = y)
loss_op = tf.reduce_mean(mse_op)
opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = opt.minimize(loss_op)

with tf.name_scope('mse'):
    tf.summary.scalar('mse', loss_op)

saver = tf.train.Saver()

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

merged = tf.summary.merge_all()
#train_writer = tf.summary.FileWriter('log/train', sess.graph)
#test_writer = tf.summary.FileWriter('log/test')

X_train_aug = np.copy(X_train)

j=0
for i in range(EPOCHS):
    X_epoch, X_valid, y_epoch, y_valid = train_test_split(X_train_aug, y_train, test_size=0.2)
    batches = split2batches(BATCH_SIZE, X_epoch, y_epoch)

    steps_per_epoch = len(X_epoch) // BATCH_SIZE

    for step in range(steps_per_epoch):
        j=j+1
        batch = batches[step]
        batch_x = batch[0]
        batch_y = batch[1]
        summary, loss  = sess.run([merged, train_op], feed_dict={x: batch_x, y: batch_y, learning_rate: rate})
        #train_writer.add_summary(summary, j)

    summary, val_loss = eval_data(X_valid, y_valid)
    #test_writer.add_summary(summary, j)
    print("EPOCH {} ...".format(i+1))
    print("Validation loss = {:.5f}".format(val_loss))

    print("rate", rate)
    print()

    
    #rate = max(rate*0.9, 0.0001)

# Evaluate on the test data
summary, test_loss = eval_data(X_test, y_test)
print("Test loss = {:.5f}".format(test_loss))


saver.save(sess, './tf_model')