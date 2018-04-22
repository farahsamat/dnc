import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np
import getopt
import sys
import os
import time
import pickle

from dnc_components.dnc import DNC
from dnc_recurrent_controller import RecurrentController
from sklearn.model_selection import train_test_split

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


def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()


if __name__ == '__main__':

    dirname = os.path.dirname(__file__)
    ckpts_dir = os.path.join(dirname, 'checkpoints')
    tb_logs_dir = os.path.join(dirname, 'logs')

    batch_size = 1
    seq_len = len(train_x)
    input_size = 128
    output_size = 50
    sequence_max_length = 128
    words_count = 15
    word_size = 10
    read_heads = 1

    learning_rate = 0.1
    momentum = 0.9

    from_checkpoint = None
    iterations = 100
    start_step = 0

    options, _ = getopt.getopt(sys.argv[1:], '', ['checkpoint=', 'iterations='])

    for opt in options:
        if opt[0] == '--checkpoint':
            from_checkpoint = opt[1]
        elif opt[0] == '--iterations':
            iterations = int(opt[1])
        elif opt[0] == '--start':
            start_step = int(opt[1])

    graph = tf.Graph()

    with graph.as_default():
        with tf.Session(graph=graph) as session:

            llprint("Building Computational Graph ... ")

            optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=momentum)

            ncomputer = DNC(
                RecurrentController,
                input_size,
                output_size,
                2 * sequence_max_length + 1,
                words_count,
                word_size,
                read_heads,
                batch_size
            )

            output, _ = ncomputer.get_outputs()
            loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=ncomputer.target_output))

            summaries = []

            gradients = optimizer.compute_gradients(loss)
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    summaries.append(tf.summary.histogram(var.name + '/grad', grad))
                    gradients[i] = (tf.clip_by_value(grad, -10, 10), var)

            apply_gradients = optimizer.apply_gradients(gradients)

            summaries.append(tf.summary.scalar("Loss", loss))

            summarize_op = tf.summary.merge(summaries)
            no_summarize = tf.no_op()

            summarizer = tf.summary.FileWriter(tb_logs_dir, session.graph)

            llprint("Done!\n")

            llprint("Initializing Variables ... ")
            session.run(tf.global_variables_initializer())
            llprint("Done!\n")

            if from_checkpoint is not None:
                llprint("Restoring Checkpoint %s ... " % (from_checkpoint))
                ncomputer.restore(session, ckpts_dir, from_checkpoint)
                llprint("Done!\n")

            last_100_losses = []
            start = 0 if start_step == 0 else start_step + 1
            end = start_step + iterations + 1

            start_time_100 = time.time()
            end_time_100 = None
            avg_100_time = 0.
            avg_counter = 0

            for i in range(iterations + 1):
                llprint("\rIteration %d/%d" % (i, iterations))
                print (' ')

                input_data = np.zeros((batch_size, 2 * seq_len + 1, input_size), dtype=np.float32)
                target_output = np.zeros((batch_size, 2 * seq_len + 1, output_size), dtype=np.float32)

                in_sequence = np.array(train_x)
                in_sequence = in_sequence.reshape(batch_size, seq_len, input_size)
                out_sequence = np.array(train_y)
                out_sequence = out_sequence.reshape(batch_size, seq_len, output_size)

                input_data[:, :seq_len, :input_size] = in_sequence
                target_output[:, seq_len + 1:, :output_size] = out_sequence

                summarize = (i % 10 == 0)
                take_checkpoint = (i != 0) and (i % iterations == 0)

                loss_value, _, summary = session.run([
                    loss,
                    apply_gradients,
                    summarize_op if summarize else no_summarize
                ], feed_dict={
                    ncomputer.input_data: input_data,
                    ncomputer.target_output: target_output,
                    ncomputer.sequence_length: 2 * seq_len + 1
                })

                last_100_losses.append(loss_value)
                summarizer.add_summary(summary, i)

                argmax_prediction = tf.argmax(output, 1)
                argmax_y = tf.argmax(ncomputer.target_output, 1)
                incorrect = tf.not_equal(argmax_prediction, argmax_y)
                misclass = tf.count_nonzero(incorrect)
                print('Misclassification:',
                      misclass.eval({ncomputer.input_data: input_data,
                                     ncomputer.target_output: target_output,
                                     ncomputer.sequence_length: 2 * seq_len + 1}), 'out of',
                      seq_len * input_size)

                hamm_loss = (tf.reduce_mean(tf.reduce_sum(misclass / output_size))) / seq_len
                print('Hamming loss:',
                      hamm_loss.eval({ncomputer.input_data: input_data,
                                      ncomputer.target_output: target_output,
                                      ncomputer.sequence_length: 2 * seq_len + 1}))

                if summarize:
                    llprint("\n\tAvg. Loss: %.4f\n" % (np.mean(last_100_losses)))

                    end_time_100 = time.time()
                    elapsed_time = (end_time_100 - start_time_100) / 60
                    avg_counter += 1
                    avg_100_time += (1. / avg_counter) * (elapsed_time - avg_100_time)
                    estimated_time = (avg_100_time * ((end - i) / 100.)) / 60.

                    print("\tAvg. 100 iterations time: %.2f minutes" % (avg_100_time))
                    print("\tApprox. time to completion: %.2f hours" % (estimated_time))

                    start_time_100 = time.time()
                    last_100_losses = []

                if take_checkpoint:
                    llprint("\nSaving Checkpoint ... "),
                    ncomputer.save(session, ckpts_dir, 'step-%d' % (i))
                    llprint("Done!\n")

            # Model evaluation
            test_seq_len = len(test_x)

            test_input = np.zeros((batch_size, 2 * test_seq_len + 1, input_size), dtype=np.float32)
            test_output = np.zeros((batch_size, 2 * test_seq_len + 1, output_size), dtype=np.float32)
            in_sequence = np.array(test_x)
            in_sequence = in_sequence.reshape(batch_size, test_seq_len, input_size)
            out_sequence = np.array(test_y)
            out_sequence = out_sequence.reshape(batch_size, test_seq_len, output_size)
            test_input[:, :test_seq_len, :input_size] = in_sequence
            test_output[:, test_seq_len + 1:, :output_size] = out_sequence

            argmax_prediction = tf.argmax(output, 1)
            argmax_y = tf.argmax(ncomputer.target_output, 1)
            incorrect = tf.not_equal(argmax_prediction, argmax_y)
            misclass = tf.count_nonzero(incorrect)
            print(' ')
            print('Model Evaluation')
            print('Misclassification:',
                  misclass.eval({ncomputer.input_data: test_input,
                                 ncomputer.target_output: test_output,
                                 ncomputer.sequence_length: 2 * test_seq_len + 1}), 'out of', test_seq_len * input_size)

            hamm_loss = (tf.reduce_mean(tf.reduce_sum(misclass / output_size))) / test_seq_len
            print('Hamming loss:',
                  hamm_loss.eval({ncomputer.input_data: test_input,
                                  ncomputer.target_output: test_output,
                                  ncomputer.sequence_length: 2 * test_seq_len + 1}))
