import tensorflow as tf
import numpy as np
import glob
from sklearn.model_selection import train_test_split

def image_pipeline(file):
    img = cv2.imread(file)
    img = cv2.resize(img, (32, 32))
    #img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img =  img[:,:, np.newaxis]
    return img

def get_data():
    X = []
    y = []

    for file in glob.glob('data/vehicles/**/*.png', recursive=True):
        X.append(image_pipeline(file))
        y.append(1)

    for file in glob.glob('data/non-vehicles/**/*.png', recursive=True):
        X.append(image_pipeline(file))
        y.append(0)

    y = np_utils.to_categorical(y, 2)

    return train_test_split(np.array(X), np.array(y), test_size=0.2)

def get_model_nvidia(): #source of the model: http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
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
    fc1 = flatten(conv2)
    # (5 * 5 * 16, 120)
    fc1_shape = (fc1.get_shape().as_list()[-1], 120)

    fc1_W = tf.Variable(tf.truncated_normal(shape=(fc1_shape), stddev=std))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc1, fc1_W) + fc1_b
    fc1 = tf.nn.relu(fc1)

    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, n_classes), stddev=std))
    fc2_b = tf.Variable(tf.zeros(n_classes))
    return tf.matmul(fc1, fc2_W) + fc2_b

X_train, X_test, y_train, y_test =  get_data()
n_train = len(X_train)
n_test = len(X_test)
n = n_train + n_test

print("size: %s, train: %s, test: %s" % (n, n_train, n_test))

model = get_model_nvidia()