from keras.datasets import cifar10
import tensorflow as tf
from tqdm import tqdm
from skimage.transform import *
from skimage.exposure import *
from sklearn.utils import shuffle
import random

keep_prob = tf.placeholder(tf.float32)


def w(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='w')


def b(shape):
    return tf.Variable(tf.zeros(shape), name='b')


def preproc(img, degree=0):
    # img = rgb2grey(img)
    for c in range(3):
        img[:,:,c] = equalize_hist(img[:,:,c])
    if degree>0:
        d = random.normalvariate(0,degree)
        img = rotate(img, d)
    return img


def DeepNet(x, features, linear, colors=1, dropout=False):
    x = tf.to_float(x)

    net = tf.nn.conv2d(x, w([5, 5, colors, features[0]]), [1, 1, 1, 1], 'VALID') + b([features[0]])
    net = tf.nn.relu(net)

    net = tf.nn.conv2d(net, w([5, 5, features[0], features[1]]), [1, 1, 1, 1], 'VALID') + b([features[1]])
    net = tf.nn.relu(net)
    if dropout: net = tf.nn.dropout(net, keep_prob)

    net = tf.nn.conv2d(net, w([5, 5, features[1], features[2]]), [1, 1, 1, 1], 'VALID') + b([features[2]])
    net = tf.nn.relu(net)
    if dropout: net = tf.nn.dropout(net, keep_prob)

    net = tf.nn.conv2d(net, w([5, 5, features[2], features[3]]), [1, 1, 1, 1], 'VALID') + b([features[3]])
    net = tf.nn.relu(net)
    if dropout: net = tf.nn.dropout(net, keep_prob)

    net = tf.nn.conv2d(net, w([5, 5, features[3], features[4]]), [1, 1, 1, 1], 'VALID') + b([features[4]])
    net = tf.nn.relu(net)
    if dropout: net = tf.nn.dropout(net, keep_prob)

    net = tf.nn.conv2d(net, w([5, 5, features[4], features[5]]), [1, 1, 1, 1], 'VALID') + b([features[5]])
    net = tf.nn.relu(net)
    if dropout: net = tf.nn.dropout(net, keep_prob)

    net = tf.nn.conv2d(net, w([5, 5, features[5], features[6]]), [1, 1, 1, 1], 'VALID') + b([features[6]])
    net = tf.nn.relu(net)
    if dropout: net = tf.nn.dropout(net, keep_prob)

    net = tf.contrib.layers.flatten(net)
    linear_size = 2048

    net = tf.matmul(net, w([linear_size, linear[0]])) + b([linear[0]])
    net = tf.nn.relu(net)
    if dropout: net = tf.nn.dropout(net, keep_prob)

    net = tf.matmul(net, w([linear[0], linear[1]])) + b([linear[1]])
    net = tf.nn.relu(net)
    if dropout: net = tf.nn.dropout(net, keep_prob)

    net = tf.matmul(net, w([linear[1], 10])) + b([10])
    return net



(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# y_train.shape is 2d, (50000, 1). While Keras is smart enough to handle this
# it's a good idea to flatten the array.
y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)

#N= 1000
#X_train = X_train[0:N,:,:,:]
#X_test = X_test[0:N,:,:,:]
#y_train = y_train[0:N]
#y_test = y_test[0:N]

with tqdm(range(len(X_train)), desc="normalize training") as pbar:
    for i in pbar:
        X_train[i] = preproc(X_train[i])

with tqdm(range(len(X_test)), desc="normalize test") as pbar:
    for i in pbar:
        X_test[i] = preproc(X_test[i])


x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))

logits = DeepNet(x, features=[16,16,32,32,64,64,128], linear=[1024,256], dropout=True, colors=3)

one_hot_y = tf.one_hot(y, 10)
rate = 0.001
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(rate, global_step, 1000, 0.95)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss = tf.reduce_mean(cross_entropy)

regularizer = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'w' in v.name])
reg_beta = 0.001
loss = tf.reduce_mean(loss + reg_beta * regularizer)

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
training_operation = optimizer.minimize(loss, global_step=global_step)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

BATCH_SIZE = 128
def evaluate(sess, X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    with tqdm(range(0, num_examples, BATCH_SIZE), desc="test") as pbar:
        for offset in pbar:
            batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
            accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
            total_accuracy += (accuracy * len(batch_x))

    return total_accuracy / num_examples


EPOCHS = 100
best_acc = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    for i in range(EPOCHS):
        X_prep, y_prep = shuffle(X_train, y_train)
        with tqdm(range(0, num_examples, BATCH_SIZE), desc="train") as pbar:
            for offset in pbar:
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: .5})

        validation_accuracy = evaluate(sess, X_test, y_test)
        best = ''
        if validation_accuracy > best_acc:
            best_acc = validation_accuracy
            # saver.save(sess, './lenet.best')
            best = '*'

        print("EPOCH {} lr= {:.5f} ValAcc = {:.3f} {}".format(
            i + 1, sess.run(learning_rate), validation_accuracy, best))