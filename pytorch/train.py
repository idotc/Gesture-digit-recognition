import h5py
import numpy as np
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from net import Net
import matplotlib.pyplot as plt

def load_dataset():
    train_dataset = h5py.File('../datasets/train_signs.h5',"r")
    train_set_x_orig = np.array(train_dataset["train_set_x"])
    train_set_y_orig = np.array(train_dataset["train_set_y"],dtype='int')

    test_dataset = h5py.File('../datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"])
    test_set_y_orig = np.array(test_dataset["test_set_y"],dtype='int')
    print (train_set_y_orig.shape)
    print (test_set_y_orig.shape)

    classes = np.array(test_dataset["list_classes"])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    print (train_set_y_orig.shape)
    print (test_set_y_orig.shape)
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    m = X.shape[1] #numbers of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1 : Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    Shuffle_X = X[:, permutation]
    Shuffle_Y = Y[:, permutation]

    # Step 2 : Partition
    num_complete_minibataches = math.floor(m / mini_batch_size)
    for k in range(0, num_complete_minibataches):
        mini_batch_X = Shuffle_X[:, k*mini_batch_size : (k+1)*mini_batch_size]
        mini_batch_Y = Shuffle_Y[:, k*mini_batch_size : (k+1)*mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size !=0: #conside the remaining value
        mini_batch_X = Shuffle_X[:, num_complete_minibataches * mini_batch_size : m]
        mini_batch_Y = Shuffle_Y[:, num_complete_minibataches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def data_process():
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
    X_train_flatten = X_train_orig.reshape((X_train_orig.shape[0],-1)).T
    X_test_flatten = X_test_orig.reshape((X_test_orig.shape[0], -1)).T
    print (X_train_flatten.shape)
    print(X_test_flatten.shape)
    X_train = X_train_flatten / 255
    X_test = X_test_flatten / 255
    C = (len(classes))
    Y_train = convert_to_one_hot(Y_train_orig, C)
    Y_test = convert_to_one_hot(Y_test_orig, C)
    return X_train, Y_train, X_test, Y_test

def train(learning_rate =0.0001, nums_epoch = 1500, minibatch_size = 32, print_cost = True):
    seed = 3
    X_train, Y_train, X_test, Y_test = data_process()
    (n_x, m)= X_train.shape
    n_y = Y_train.shape[0]
    costs = []

    net = Net(12288, 25, 16, 6)
    print (net)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    #loss_fuc = nn.CrossEntropyLoss()
    loss_fuc = nn.MultiLabelSoftMarginLoss()

    for epoch in range(1500):
        epoch_cost = 0
        epoch_accracy = 0
        num_batches = int(m / minibatch_size)
        seed = seed + 1
        minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

        for minibatch in minibatches:
            (minibatchX, minibatchY) = minibatch
            minibatchX = minibatchX.astype(np.float32).T
            minibatchY = minibatchY.astype(np.float32).T
            b_x = Variable(torch.from_numpy(minibatchX))
            b_y = Variable(torch.from_numpy(minibatchY))

            output = net(b_x)
            minibatch_cost = loss_fuc(output, b_y)
            optimizer.zero_grad()
            minibatch_cost.backward()
            optimizer.step()
            #print(b_y)
            #print(torch.max(b_y, 1))
            #print(torch.max(b_y, 1)[1].data.squeeze())
            correct_prediction = sum(torch.max(output, 1)[1].data.squeeze() == torch.max(b_y, 1)[1].data.squeeze())
            epoch_accracy += correct_prediction / minibatch_size / len(minibatches)
            epoch_cost += minibatch_cost / len(minibatches)

        if print_cost == True and epoch % 100 ==0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
                print ("Traing Acc. after epoch %i: %f" % (epoch, epoch_accracy))
                X_test_tensor = X_test.astype(np.float32).T
                Y_test_tensor = Y_test.astype(np.float32).T
                X_test_tensor = Variable(torch.from_numpy(X_test_tensor))
                Y_test_tensor = Variable(torch.from_numpy(Y_test_tensor))
                test_output = net(X_test_tensor)
                correct_prediction = sum(torch.max(test_output, 1)[1].data.squeeze() == torch.max(Y_test_tensor, 1)[1].data.squeeze())
                #correct_prediction = sum(torch.max(test_output) == torch.max(Y_test_tensor))
                correct_prediction = correct_prediction / test_output.size(0)
                print ("Traing Acc. after epoch %i: %f" % (epoch, correct_prediction))

        if print_cost == True and epoch%5 == 0:
                costs.append(epoch_cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()


if __name__ == "__main__":
    train()
#load_dataset()
#random_mini_batches([2,2,3,1,4,5,6,1],[1,2,3,4,5,6,7,4],mini_batch_size = 3)
