import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
from tensorflow.python.framework import ops
import utils
import time
import argparse
import math
import os
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--window_size', type=int, default=32)
    parser.add_argument('--sampling', type=str, default="p60")
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--n_hidden', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--export_path', type=str, default="lstmTrainResults")
    parser.add_argument('--ploting', type=str, default="y")
    args = parser.parse_args()
    run(args)

def run(args):
    dataTrain, dataTest, maxData = utils.getData("../Database/vViento.csv", args.sampling, 80)

    dataTrain = dataTrain / maxData
    dataTrain = dataTrain.tolist()

    dataTest = dataTest / maxData
    dataTest = dataTest.tolist()


    rollback = args.window_size

    x_data = []
    y_data = []

    for i in xrange(len(dataTrain) - rollback):
        x_data.append(dataTrain[i:i+rollback])
        y_data.append(dataTrain[i+rollback])

    def RNN(_X, _istate, _weights, _biases):

        # input shape: (batch_size, n_steps, n_input)
        _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
        # Reshape to prepare input to hidden activation
        _X = tf.reshape(_X, [-1, n_input])  # (n_steps*batch_size, n_input)
        # Linear activation
        _X = tf.matmul(_X, _weights['hidden']) + _biases['hidden']

        # Define a lstm cell with tensorflow
        lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
        # Split data because rnn cell needs a list of inputs for the RNN inner loop
        _X = tf.split(0, window_size, _X)  # n_steps * (batch_size, n_hidden)

        outputs, states = rnn.rnn(lstm_cell, _X, initial_state=_istate)

        return tf.matmul(outputs[-1], _weights['out']) + _biases['out']


    # Network Parameters
    n_input = 1
    n_outputs = 1
    num_epoch = args.num_epoch
    window_size = rollback
    batch_size = args.batch_size
    n_hidden = args.n_hidden
    lr= args.lr

    print "size_dataTrain:", len(dataTrain)
    print "size_dataTest:", len(dataTest)
    print "num_epoch:", num_epoch
    print "window_size:", window_size
    print "batch_size:", batch_size
    print "n_hidden:", n_hidden
    print "lr:", lr

    x = tf.placeholder("float", [None, window_size, n_input], name="input_placeholder_x")
    istate = tf.placeholder("float", [None, 2*n_hidden], name="cell_estate")
    y = tf.placeholder("float", [None, n_input], name="input_placeholder_y")

    # Define weights
    weights = {
        'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])),
        'out': tf.Variable(tf.random_normal([n_hidden, n_outputs]))
    }
    biases = {
        'hidden': tf.Variable(tf.random_normal([n_hidden])),
        'out': tf.Variable(tf.random_normal([n_outputs]))
    }

    pred = RNN(x, istate, weights, biases)

    cost = tf.reduce_mean(tf.pow(pred - y, 2))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)  # Adam Optimizer

    saver = tf.train.Saver()

    init = tf.initialize_all_variables()

    cost_training = list()

    with tf.Session() as sess:
        sess.run(init)
        print "----------------------"
        print "   Start training...  "
        print "----------------------"

        t = time.time()
        for epoch in xrange(num_epoch):
            for jj in xrange(len(x_data) / batch_size):
                batch_xs = x_data[jj*batch_size : jj*batch_size+batch_size]
                batch_ys = y_data[jj*batch_size : jj*batch_size+batch_size]
                _, cost_train = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys, istate: np.zeros((batch_size, 2*n_hidden))})

            cost_training.append(cost_train)
            if epoch % 10 == 0 :
                print "Epoch: ", epoch
                print "  Cost:", cost_train

        t = time.time() - t


        print "----------------------"
        print "   End training...    "
        print "----------------------"

        print "----------------------"
        print "   Start prediction..."
        print "----------------------"

        prediction = list()
        BIAS = list()
        MAE = list()
        MSE = list()
        RMSE = list()

        N = len(dataTest)
        print N

        test = dataTrain[len(dataTrain)-window_size:] + dataTest

        for i in range(N):
            x_data = test[i:i+window_size]
            output = sess.run(pred, feed_dict={x: [x_data], istate: np.zeros((1, 2*n_hidden))})
            prediction.append(output[0][0]*maxData)

            predData = output[0][0]
            realData = test[i+window_size][0]

            # Errors calculation
            bias = realData - predData
            mae = abs(realData - predData)
            mse = math.pow(realData - predData, 2)
            rmse = math.sqrt(math.pow(realData - predData, 2))

            BIAS.append(bias)
            MAE.append(mae)
            MSE.append(mse)
            RMSE.append(rmse)

        print "----------------------"
        print "   Save results...    "
        print "----------------------"

        print "Train time", t
        eBIAS = np.sum(BIAS)/N
        print "BIAS:", eBIAS
        eMAE = np.sum(MAE)/N
        print "MAE:", eMAE
        eMSE = np.sum(MSE)/N
        print "MSE:", eMSE
        eRMSE = np.sum(RMSE)/N
        print "RMSE:", eRMSE
        eSDE = math.sqrt(np.sum(np.power((BIAS - eBIAS), 2))/N)
        print "SDE:", eSDE

        if os.path.exists(args.export_path)==0: os.mkdir(args.export_path)

        n = input("Save?: ")
        if n == 1:
            print "Saved!!"
            saver.save(sess, args.export_path + "/best_model.ckpt")

        f1 = open(args.export_path + '/errors.txt', 'a')
        f1.write(str(eBIAS) + ";" + str(eMAE) + ";" + str(eMSE) + ";" + (str(eRMSE) + ";" + str(eSDE) + ";" + str(t) + "\n"))
        f1.close()

        if args.ploting == "y":
            dataTest = np.array(dataTest) * maxData
            utils.saveTrainingResults([num_epoch, window_size, batch_size, n_hidden, lr], cost_training,
                             prediction, dataTest, args.export_path)

    sess.close()
    ops.reset_default_graph()

if __name__ == '__main__':
    main()
