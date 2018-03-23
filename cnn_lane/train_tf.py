import tensorflow as tf
from tensorflow.contrib.layers import flatten

import numpy as np
import glob
from sklearn.model_selection import train_test_split
#import cv2
import math
from PIL import Image

EPOCHS = 20
BATCH_SIZE = 10
rate = 0.0001

def image_pipeline(file):
    img = Image.open(file)#.convert('LA')
    img.load()
    #img = img.resize((64, 64), Image.ANTIALIAS)

    data = np.asarray(img, dtype="float")
    data = np.dot(data[...,:3], [0.299, 0.587, 0.114]) # to gray
    data = data / 255 # normalization

    h, w = data.shape
    data = data[h/2:h, :]

    data = data[:,:, np.newaxis]
    return data

def get_data():
    X = []
    y = []

    txt = np.loadtxt("/home/kvasnyj/Dropbox/carla/cnn_lane/data/data.txt", delimiter=";")
    #txt = txt[txt[:, 6]<800]

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
    conv1 = tf.layers.conv2d(inputs=x, filters=3, kernel_size=[5, 5], padding="same", activation=tf.nn.elu)
    conv2 = tf.layers.conv2d(inputs=conv1, filters=24, kernel_size=[5, 5], padding="same", activation=tf.nn.elu)
    conv3 = tf.layers.conv2d(inputs=conv2, filters=36, kernel_size=[5, 5], padding="same", activation=tf.nn.elu)
    conv4 = tf.layers.conv2d(inputs=conv3, filters=48, kernel_size=[5, 5], padding="same", activation=tf.nn.elu)
    conv5 = tf.layers.conv2d(inputs=conv4, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.elu)
    conv6 = tf.layers.conv2d(inputs=conv5, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.elu)

    fc0 = flatten(conv6)
    fc0d = tf.layers.dropout(inputs=fc0, rate=0.2)

    fc1 = tf.layers.dense(inputs=fc0d, units=1164, activation=tf.nn.elu)
    fc1d = tf.layers.dropout(inputs=fc1, rate=0.2)

    fc2 = tf.layers.dense(inputs=fc1d, units=100, activation=tf.nn.elu)
    fc2d = tf.layers.dropout(inputs=fc2, rate=0.2)

    fc3 = tf.layers.dense(inputs=fc2d, units=50, activation=tf.nn.elu)
    fc4d = tf.layers.dropout(inputs=fc3, rate=0.5)

    fc4 = tf.layers.dense(inputs=fc3d, units=10, activation=tf.nn.elu)
    fc4d = tf.layers.dropout(inputs=fc4, rate=0.5)

    fc5 = tf.layers.dense(inputs=fc4d, units=6)

  return fc5

def eval_data(X_valid, y_valid):
    num_examples = len(X_valid)
    summary, loss, acc = sess.run([merged, loss_op, accuracy_op], feed_dict={x: X_valid, y: y_valid})

    return summary, loss, acc

X_train, X_test, y_train, y_test =  get_data()
n_train = len(X_train)
n_test = len(X_test)
n = n_train + n_test

print("size: %s, train: %s, test: %s" % (n, n_train, n_test))

image_shape = X_train[0].shape
x = tf.placeholder(tf.float32, (None, image_shape[0], image_shape[1], 1))
y = tf.placeholder(tf.float32, (None, 6))
learning_rate = tf.placeholder(tf.float32, shape=[])

model = get_model_nvidia(x)

mse = tf.losses.mean_squared_error(predictions = model, labels = y)
loss_op = tf.reduce_mean(mse)
correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = opt.minimize(loss_op)

with tf.name_scope('mse'):
    tf.summary.scalar('mse', loss_op)

with tf.name_scope('accuracy'):
    tf.summary.scalar('accuracy', accuracy_op)

saver = tf.train.Saver()

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('log/train', sess.graph)
test_writer = tf.summary.FileWriter('log/test')

j=0
for i in range(EPOCHS):
    X_epoch, X_valid, y_epoch, y_valid = train_test_split(X_train, y_train, test_size=0.2)
    batches = split2batches(BATCH_SIZE, X_epoch, y_epoch)

    steps_per_epoch = len(X_epoch) // BATCH_SIZE

    for step in range(steps_per_epoch):
        j=j+1
        batch = batches[step]
        batch_x = batch[0]
        batch_y = batch[1]
        summary, loss  = sess.run([merged, train_op], feed_dict={x: batch_x, y: batch_y, learning_rate: rate})
        train_writer.add_summary(summary, j)

    summary, val_loss, val_acc = eval_data(X_valid, y_valid)
    test_writer.add_summary(summary, j)
    print("EPOCH {} ...".format(i+1))
    print("Validation loss = {:.3f}".format(val_loss))
    print("Validation accuracy = {:.3f}".format(val_acc))

    print("rate", rate)
    print()

# Evaluate on the test data
summary, test_loss, test_acc = eval_data(X_test, y_test)
print("Test loss = {:.3f}".format(test_loss))
print("Test accuracy = {:.3f}".format(test_acc))

saver.save(sess, './tf_model')