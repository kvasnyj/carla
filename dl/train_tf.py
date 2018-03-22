import tensorflow as tf
from tensorflow.contrib.layers import flatten
import numpy as np
import glob
from sklearn.model_selection import train_test_split
#import cv2
import math

n_classes = 2
EPOCHS = 20
BATCH_SIZE = 100
rate = 0.001

from PIL import Image
def image_pipeline_pil(file):
    img = Image.open(file)#.convert('LA')
    img.load()
    #img = img.resize((32, 32), Image.ANTIALIAS)
    data = np.asarray(img, dtype="int32")
    data = np.dot(data[...,:3], [0.299, 0.587, 0.114])
    data = data[:,:, np.newaxis]
    return data

def image_pipeline(file):
    img = cv2.imread(file)
    img = cv2.resize(img, (32, 32))
    #img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img =  img[:,:, np.newaxis]
    return img

def array2classifier(M, array):
    N = len(array)
    resultArray = np.zeros((N, M), float)
    resultArray[np.arange(N), array] = 1.
    return resultArray

def get_data():
    X = []
    y = []

    for file in glob.glob('data/vehicles/**/*.png', recursive=True):
        X.append(image_pipeline_pil(file))
        y.append(1)

    for file in glob.glob('data/non-vehicles/**/*.png', recursive=True):
        X.append(image_pipeline_pil(file))
        y.append(0)

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

def conv_layer(input, size_in, size_out, name="conv"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
        act = tf.nn.relu(conv + b)
        act = tf.nn.relu(conv + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
    return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def fc_layer(input, size_in, size_out, name="fc"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        act = tf.matmul(input, w) + b
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
    return act

def get_model_lenet(x):
    x = tf.reshape(x, (-1, 64, 64, 1))

    conv1 = conv_layer(x, 1, 6, "conv1")

    conv2 = conv_layer(conv1, 6,  16, "conv2")

    fc1 = flatten(conv2)

    fc1 = fc_layer(fc1, fc1.get_shape().as_list()[-1], 120, "fc1")
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, 0.5)
    fc2 = fc_layer(fc1, 120, n_classes, "fc2")
    return fc2

def get_model_lenet_layer(x):
    x = tf.reshape(x, (-1, 64, 64, 1))

    conv1 = tf.layers.conv2d(
      inputs=x,
      filters=6,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=16,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    fc0 = flatten(pool2)
    fc1 = tf.layers.dense(inputs=fc0, units=120, activation=tf.nn.relu)
    fc1d = tf.layers.dropout(inputs=fc1, rate=0.4)
    fc2 = tf.layers.dense(inputs=fc1d, units=n_classes)
    return fc2

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
y = tf.placeholder(tf.float32, (None, n_classes))
learning_rate = tf.placeholder(tf.float32, shape=[])

model = get_model_lenet_layer(x)

xent = tf.nn.softmax_cross_entropy_with_logits(logits = model, labels = y)
loss_op = tf.reduce_mean(xent)
correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = opt.minimize(loss_op)

with tf.name_scope('xent'):
    tf.summary.scalar('xent', loss_op)

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
        batch_y = array2classifier(n_classes, batch[1])
        summary, loss  = sess.run([merged, train_op], feed_dict={x: batch_x, y: batch_y, learning_rate: rate})
        train_writer.add_summary(summary, j)

    summary, val_loss, val_acc = eval_data(X_valid, array2classifier(n_classes, y_valid))
    test_writer.add_summary(summary, j)
    print("EPOCH {} ...".format(i+1))
    print("Validation loss = {:.3f}".format(val_loss))
    print("Validation accuracy = {:.3f}".format(val_acc))

    print("rate", rate)
    #rate = max(rate*0.5, 0.00001)

    print()

# Evaluate on the test data
summary, test_loss, test_acc = eval_data(X_test, array2classifier(n_classes, y_test))
print("Test loss = {:.3f}".format(test_loss))
print("Test accuracy = {:.3f}".format(test_acc))

saver.save(sess, './tf_model')