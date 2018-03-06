import tensorflow as tf
from tensorflow.contrib.layers import flatten
import numpy as np
import glob
from sklearn.model_selection import train_test_split
import cv2
import math

n_classes = 2
std = 0.1
EPOCHS = 20
BATCH_SIZE = 100
rate = 0.0001


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
        X.append(image_pipeline(file))
        y.append(1)

    for file in glob.glob('data/non-vehicles/**/*.png', recursive=True):
        X.append(image_pipeline(file))
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

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def fc_layer(input, weight_shape, bias_shape, name):
    weight_init = tf.truncated_normal(shape=(fc1_shape), stddev=std)
    bias_init = tf.constant_initializer(value=0)
    W = tf.get_variable("W", weight_shape, initializer=weight_init)
    b = tf.get_variable("b", bias_shape, initializer=bias_init)
    fc = tf.add(tf.matmul(input, W), b, name=name)
    fc = tf.nn.relu(fc)

    with tf.name_scope(name):
    # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = weight_variable(weight_shape)
            variable_summaries(W)
        with tf.name_scope('biases'):
            biases = bias_variable(bias_shape)
            variable_summaries(b)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, W) + b
            tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)

    return fc

def get_model_lenet(x): #source of the model: http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    # 28x28x6
    x = tf.reshape(x, (-1, 32, 32, 1))
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), stddev=std))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    conv1 = tf.nn.relu(conv1)

    # 14x14x6
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # 10x10x16
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), stddev=std))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    conv2 = tf.nn.relu(conv2)

    # 5x5x16
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten
    fc1_shape = (fc1.get_shape().as_list()[-1], 120)
    fc1 = fc_layer(flatten(conv2), fc1_shape, 120, "fc1")
    #fc1 = tf.nn.dropout(fc1, 0.5)
    fc2 = fc_layer(fc1, (120, n_classes), n_classes, "fc2")
    return fc2

def eval_data(X_valid, y_valid):
    num_examples = len(X_valid)
    summary, loss, acc = sess.run([merged, loss_op, accuracy_op], feed_dict={x: batch_x, y: batch_y})

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

model = get_model_lenet(x)

with tf.name_scope('cross_entropy'):
    with tf.name_scope('total'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = model, labels = y)
tf.summary.scalar('cross_entropy', cross_entropy)

loss_op = tf.reduce_mean(entropy)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    with tf.name_scope('accuracy'):
        accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
with tf.name_scope('train'):
    train_op = opt.minimize(loss_op)

# Merge all the summaries and write them out to /tmp/mnist_logs (by default)
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('log/train', sess.graph)
test_writer = tf.summary.FileWriter('log//test')

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

j = 0
for i in range(EPOCHS):
    X_epoch, X_valid, y_epoch, y_valid = train_test_split(X_train, y_train, test_size=0.2)
    batches = split2batches(BATCH_SIZE, X_epoch, y_epoch)

    steps_per_epoch = len(X_epoch) // BATCH_SIZE

    for step in range(steps_per_epoch):
        j = j + 1
        batch = batches[step]
        batch_x = batch[0]
        batch_y = array2classifier(n_classes, batch[1])
        summary, _  = sess.run([merged, train_step], train_op, feed_dict={x: batch_x, y: batch_y, learning_rate: rate})
        train_writer.add_summary(summary, j)

    summary, val_loss, val_acc = eval_data(X_valid, y_valid)
    test_writer.add_summary(summary, j)
    print("EPOCH {} ...".format(i+1))
    print("Validation loss = {:.3f}".format(val_loss))
    print("Validation accuracy = {:.3f}".format(val_acc))

    print("rate", rate)
    #rate = max(rate*0.5, 0.00001)

    print()

# Evaluate on the test data
test_loss, test_acc = eval_data(X_test, array2classifier(n_classes, y_test))
print("Test loss = {:.3f}".format(test_loss))
print("Test accuracy = {:.3f}".format(test_acc))