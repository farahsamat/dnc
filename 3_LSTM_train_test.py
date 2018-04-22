import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import sys
import time
import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

tf.reset_default_graph()

# Load data
print ("Loading data...")
with open('/data/embedding.pickle', 'rb') as f:
    input = pickle.load(f)
print (len(input))
with open('/data/y_raw.pickle', 'rb') as f:
    output = pickle.load(f)
print (len(output))
print("Data loaded\n")


# Split data
train_x, test_x = train_test_split(input, test_size=0.1, random_state=42)
train_y, test_y = train_test_split(output, test_size=0.1, random_state=42)

##=======================================##
dirname = os.path.dirname(__file__)
tb_logs_dir = os.path.join(dirname, 'Logs')
hm_epochs = 100
output_size = 50
batch_size = 1
input_size = 128
seq_len = len(train_x)

rnn_size = 64

learning_rate = 0.001
momentum = 0.9
start_step = 0

x = tf.placeholder('float', [None, None, input_size])
y = tf.placeholder('float')


def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()


def recurrent_neural_network(x):
    layer = {'weights': tf.Variable(tf.random_normal([rnn_size, output_size])),
             'biases': tf.Variable(tf.random_normal([output_size]))}

    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, input_size])
    x = tf.split(x, 1, 0)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size, state_is_tuple=True)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.nn.sigmoid(tf.matmul(outputs[-1], layer['weights']) + layer['biases'])

    return output


def train_neural_network(x):
    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=momentum).minimize(cost)

    with tf.Session() as sess:
        llprint("Building Computational Graph ... ")
        summaries = []
        summaries.append(tf.summary.scalar("Loss", cost))
        summarize_op = tf.summary.merge(summaries)
        no_summarize = tf.no_op()
        summarizer = tf.summary.FileWriter(tb_logs_dir, sess.graph)
        llprint("Done!\n")

        llprint("Initializing variables...")
        sess.run(tf.global_variables_initializer())
        llprint("Done!\n")

        last_100_losses = []
        start = 0 if start_step == 0 else start_step + 1
        end = start_step + hm_epochs + 1

        start_time_100 = time.time()
        end_time_100 = None
        avg_100_time = 0.
        avg_counter = 0

        for epoch in range(hm_epochs + 1):
            llprint("\rIteration %d/%d" % (epoch, hm_epochs))
            summarize = (epoch % 10 == 0)
            epoch_loss = 0

            epoch_x, epoch_y = np.array(train_x), np.array(train_y)
            epoch_x = epoch_x.reshape((-1, 1, input_size))

            _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
            epoch_loss += c

            last_100_losses.append(c)
            if summarize:
                llprint("\n\tAvg. Loss: %.4f\n" % (np.mean(last_100_losses)))
                print(last_100_losses)

                end_time_100 = time.time()
                elapsed_time = (end_time_100 - start_time_100) / 60
                avg_counter += 1
                avg_100_time += (1. / avg_counter) * (elapsed_time - avg_100_time)
                estimated_time = (avg_100_time * ((end - epoch) / 100.)) / 60.

                print("\tAvg. 100 iterations time: %.2f minutes" % (avg_100_time))
                print("\tApprox. time to completion: %.2f hours" % (estimated_time))

                start_time_100 = time.time()
                last_100_losses = []

# Model eval
        argmax_prediction = tf.argmax(prediction, 1)
        argmax_y = tf.argmax(y, 1)
        incorrect = tf.not_equal(argmax_prediction, argmax_y)
        misclass = tf.count_nonzero(incorrect)
        hamm_loss = ((tf.reduce_sum(misclass / output_size))) / len((test_x))
        print('Hamming loss:', hamm_loss.eval({x: np.array(test_x).reshape((-1, len(test_x), input_size)), y: test_y}))


train_neural_network(x)

print("Done")



