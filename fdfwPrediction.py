# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import os
import time
import utils
import matplotlib.pyplot as plt
import argparse
import math
from tensorflow.python.framework import ops

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--window_size', type=int, default=32)
    parser.add_argument('--sampling', type=str, default="p60")
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--n_hidden', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--export_path', type=str, default="fdfwTrainResults")
    parser.add_argument('--ploting', type=str, default="yes")
    args = parser.parse_args()
    run(args)

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.1))
 def run(args):
    # Network Parameters
    num_epoch = args.num_epoch
    window_size = args.window_size
    batch_size = args.batch_size
    lr = args.lr
    n_hidden = args.n_hidden

    dataTrain, dataTest, maxData = utils.getData2("../Database/vViento.csv", args.sampling, 80)

    print len(dataTrain), len(dataTest)

    dataTrain = dataTrain / maxData
    dataTrain = dataTrain.tolist()

    dataTest = dataTest / maxData
    dataTest = dataTest.tolist()

    print "size_dataTrain:", len(dataTrain)
    print "size_dataTest:", len(dataTest)
    print "num_epoch:", num_epoch
    print "window_size:", window_size
    print "batch_size:", batch_size
    print "n_hidden:", n_hidden
    print "lr:", lr

    x_data = []
    y_data = []

    for i in xrange(len(dataTrain) - window_size):
        x_data.append(dataTrain[i:i+window_size])
        y_data.append(dataTrain[i+window_size])

    # Model
    x = tf.placeholder("float", [None, window_size])
    y_ = tf.placeholder("float", [batch_size, 1])

    W_hidden = init_weights([window_size, n_hidden])
    b_hidden = init_weights([n_hidden])

    W_output = init_weights([n_hidden, 1])
    b_output = init_weights([1])

    y_hidden = tf.sigmoid(tf.matmul(x, W_hidden) + b_hidden)

    pred = tf.sigmoid(tf.matmul(y_hidden, W_output) + b_output)

    cost = tf.reduce_sum(tf.square(y_ - pred))

    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost)

    saver = tf.train.Saver()

    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    cost_training = list()

    print "----------------------"
    print "   Start training...  "
    print "----------------------"

    t = time.time()

    for epoch in xrange(num_epoch):
        for jj in xrange(len(x_data) / batch_size):
            batch_xs = x_data[jj*batch_size : jj*batch_size+batch_size]
            batch_ys = y_data[jj*batch_size : jj*batch_size+batch_size]
            _, cost_train = sess.run([optimizer, cost], feed_dict={x: np.atleast_2d(batch_xs), y_: np.atleast_2d(batch_ys).T})

        cost_training.append(cost_train)
        if epoch % 10 == 0:
            print("Epoch: %3d" % (epoch))
            print("   Cost: %4.6f" % cost_train)
            print "----------------------------------"

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

    test = dataTrain[len(dataTrain)-window_size:] + dataTest

    for i in range(N):
        x_data = test[i:i+window_size]
        output = sess.run(pred, feed_dict={x: np.atleast_2d(x_data)})
        prediction.append(output[0][0]*maxData)

        predData = output[0][0]
        realData = test[i+window_size]

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

    n = input("Do you want to save the model?(1->Yes/0->No): ")
    if n == 1:
        print "Saved!!"
        saver.save(sess, args.export_path + "/best_model.ckpt")


    f1 = open(args.export_path + '/errors.txt', 'a')
    f1.write(str(eBIAS) + ";" + str(eMAE) + ";" + str(eMSE) + ";" + (str(eRMSE) + ";" + str(eSDE) + ";" + str(t) + "\n"))
    f1.close()

    if args.ploting == "yes":
            dataTest = np.array(dataTest) * maxData
            utils.saveTrainingResults([num_epoch, window_size, batch_size, n_hidden, lr], cost_training,
                             prediction, dataTest, args.export_path)
    sess.close()
    ops.reset_default_graph()

if __name__ == '__main__':
    main()
