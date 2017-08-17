##summary:

#3.1.17 : - added preplexity calculation
#          (was added as an argument that returns from lossFunc)
#         - created a plot function (PREP & #N will be as lists)
#         - NEED TO ADD ITERATIONS FOR CALC PREP


# IMPORTS:
import numpy as np
import math
import sys
import pickle
# HC: added Data bases
# from plotting_for_project import sample_perplexity_DB, train_perplexity_DB, loss_DB, out_DB

#### prepare input
input_N = int(input('Insert N (hidden layer size):  '))  # HC - added in order to summon any run of the script with different N
log_file = open('log_file.txt', 'a')
# f = open('input.txt','w')   # read data
# a = input('is python good?')
# f.write('answer:'+str(a))
# f.close()

# Train set:
data = open('shkpsr_RAW_TXT.txt', 'r').read()  # should be simple plain text file HC - this is the train set
# data = open('ML_project/shkpsr_RAW_TXT.txt', 'r').read() # should be simple plain text file # TODO - change to the following
chars = list(set(data))  #  vocabulary
data_size, vocab_size = len(data), len(chars)
# print ('data has {} characters, {} unique.'.format(int(data_size), int(vocab_size))) #% (int(data_size), int(vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }    # vocabulary
ix_to_char = { i:ch for i,ch in enumerate(chars) }    # index

# print (char_to_ix)

# Test set:     HC - added for test set: (train set / test set ratio ~ 80% / 20%)
test_set = open('shkspr_test_set.txt', 'r').read()
#  same process for test set:
test_chars = list(set(test_set))  # vocabulary  # TODO - need to understand if I should use the original char_to_ix and ix_to_char or generate a new one for test set
test_data_size, test_vocab_size = len(test_set), len(test_chars)
# print ('data has {} characters, {} unique.'.format(int(data_size), int(vocab_size))) #% (int(data_size), int(vocab_size))
# char_to_ix = { ch:i for i,ch in enumerate(chars) }    # vocabulary
# ix_to_char = { i:ch for i,ch in enumerate(chars) }    # index

####prepare network parameters

# hyperparameters
hidden_size = input_N  # size of hidden layer of neurons  # HC - given as input from command line
# hidden_size = 100  # size of hidden layer of neurons
seq_length = 25  # number of steps to unroll the RNN for
learning_rate = 1e-1
threshold = 0.00001  # HC - declare threshold size for gradient descent

# model parameters
Wxh = np.random.randn(hidden_size, vocab_size)*0.01  # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01  # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01  # hidden to output
bh = np.zeros((hidden_size, 1))  # hidden bias
by = np.zeros((vocab_size, 1))  # output bias


########  loss functions ########
def lossFun(inputs, targets, hprev):
    """
    inputs,targets are both list of integers.
    hprev is Hx1 array of initial hidden state
    returns the loss, gradients on model parameters, and last hidden state
    """
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0
    counter = 0  #HC - added 3.1.17
    # forward pass
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size,1))  # encode in 1-of-k representation
        xs[t][inputs[t]] = 1
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh)  # hidden state
        #Whh*hs-->Whh*y_syn*hs; y_syn[t+1]=MishaModel(y_syn[t],tau,U,hs) xe*xg(t)
        ys[t] = np.dot(Why, hs[t]) + by  # unnormalized log probabilities for next chars
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
        loss += -np.log(ps[t][targets[t],0])  # softmax (cross-entropy loss)
        counter += 1  # HC - added 3.1.17
        # backward pass: compute gradients going backwards
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])
    for t in reversed(range(len(inputs))):
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        dh = np.dot(Why.T, dy) + dhnext # backprop into h
        dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
        dbh += dhraw
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t-1].T)
        dhnext = np.dot(Whh.T, dhraw)
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients

    perplexity = 2**(loss/counter)  # HC - was added as the lossFun function perplexity

    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1], ys, perplexity  # HC - also returning perplexity


def test_loss_function(test_inputs, test_targets, hprev):  # HC - added test loss function:  # TODO - implement
    '''
    A modulated version of the original lossFunc.
    Doesn't change the net weights / elements.  Only calculates loss and perplexity for test set.
    inputs,targets are both list of integers.
    hprev is Hx1 array of initial hidden state
    returns the loss, and the perplexity of the test set under the current net state.
    '''
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    test_set_loss = 0
    counter = 0

    # forward pass
    for t in range(len(test_inputs)):
        xs[t] = np.zeros((test_vocab_size,1))  # encode in 1-of-k representation
        xs[t][test_inputs[t]] = 1
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh)  # hidden state
        #Whh*hs-->Whh*y_syn*hs; y_syn[t+1]=MishaModel(y_syn[t],tau,U,hs) xe*xg(t)
        ys[t] = np.dot(Why, hs[t]) + by  # un normalized log probabilities for next chars
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))  # probabilities for next chars
        test_set_loss += -np.log(ps[t][test_targets[t], 0])  # softmax (cross-entropy test_set_loss)
        counter += 1  # counter for perplexity calculation

    # backward pass: compute gradients going backwards
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])
    for t in reversed(range(len(test_inputs))):
        dy = np.copy(ps[t])
        dy[test_targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        dh = np.dot(Why.T, dy) + dhnext # backprop into h
        dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
        dbh += dhraw
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t-1].T)
        dhnext = np.dot(Whh.T, dhraw)

    # clipping:
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients

    # perplexity calculation:
    test_set_perplexity = 2**(test_set_loss/counter)

    return test_set_loss, test_set_perplexity  # HC - return arguments were defined by Gal TODO - might need to return extra arguments



######## sample function  #########
def sample(h, seed_ix):
    """
    sample a sequence of integers from the model
    h is memory state, seed_ix is seed letter for first time step
    """
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    p = []
    for t in range(200): # samples with a 200 characters long text (samples a subtext from the original text)
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y = np.dot(Why, h) + by
        p = np.exp(y) / np.sum(np.exp(y))  # softmax
        # aux_perp += -np.log2(float(p[t]))
        # choose randomly (with accordance to it's probability) one of the chars from the vocabulary :
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)
    # HC - calculate sample function perplexity:  (append it to the return arguments)
    aux_perp = float(0)
    for i in range(len(p)):
        aux_perp += -np.log2(float(p[i]))
    perplexity_ = 2**(aux_perp / len(p))
    return ixes, perplexity_



# initialization
n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log2(1.0/vocab_size)*seq_length # loss at iteration 0

# HC - added initial perplexity value (in order to obtain smooth perplexity at the end of the process):
init_p = 1.0 / vocab_size  # uniformed probability # TODO - is it correct assumption
init_softmax = np.exp(init_p) / (vocab_size * np.exp(init_p))
training_perplexity_ = 2**((-1.0) * np.log2(init_softmax))  # init perplexity at iteration 0

# HC - init smooth loss and smooth perplexity for test set:
test_smooth_loss = -np.log2(1.0/test_vocab_size)*test_data_size
init_test_p = 1.0 / test_vocab_size
init_test_softmax = np.exp(init_test_p) / (test_vocab_size * np.exp(init_test_p))
test_perplexity_ = 2**((-1.0) * np.log2(init_test_softmax))  # init perplexity at iteration 0


# loss=80
Loss_arr = []
Out = []
sample_perplexity = []
training_perplexity = []
test_loss = []
test_perplexity = []
# running = 0

while True:
    # print(running)
    # running += 1
    # prepare inputs (we're sweeping from left to right in steps seq_length long)
    if p+seq_length+1 >= len(data) or n == 0:
        hprev = np.zeros((hidden_size,1)) # reset RNN memory
        p = 0 # go from start of data
    inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
    targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

    # sample from the model now and then (and in each 100 iterations use again)
    if n % 100 == 0:
        sample_ix, sample_perplexity_iteration = sample(hprev, inputs[0])
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        sample_perplexity.append(sample_perplexity_iteration)
        # print ('----\n %s \n----' % (txt, ))

    # forward seq_length characters through the net and fetch gradient
    loss, dWxh, dWhh, dWhy, dbh, dby, hprev, y, training_perplexity_iteration = lossFun(inputs, targets, hprev)  #HC - added train_prep

    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    # if n % 100 == 0: print ('iter %d, loss: %f' % (n, smooth_loss)) # print progress

    # HC - modified the perplexity to smooth perplexity (same method used in loss):
    training_perplexity_ = training_perplexity_ * 0.999 + training_perplexity_iteration * 0.001
    training_perplexity.append(training_perplexity_)


    # perform parameter update with Adagrad
    for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                [dWxh, dWhh, dWhy, dbh, dby],
                                [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update
    # print (y)
    p += seq_length # move data pointer
    n += 1  # iteration counter

    # HC - added for test set:
    # calculate loss value for test set:
    test_inputs = [char_to_ix[ch] for ch in test_set[0:len(test_set)-1]]  # TODO - might need to return to test_set[0:p+seq_length]]
    test_targets = [char_to_ix[ch] for ch in test_set[1:len(test_set)]]
    test_loss_iteration, test_perplexity_iteration = test_loss_function(test_inputs, test_targets, hprev)
    test_smooth_loss = test_smooth_loss * 0.999 + test_loss_iteration * 0.001
    test_loss.append(test_smooth_loss)
    # calculate perplexity fot test set:
    test_perplexity_ = test_perplexity_ * 0.999 + test_perplexity_iteration * 0.001
    test_perplexity.append(test_perplexity_)


    # insert Loss and current char (in txt) to the log memory list:
    Loss_arr.append(smooth_loss) # HC -  changed the argument from loss to smooth loss.
    Out.append(txt)
    # HC - stop condition - abs(delta) < T       - added by Gal
    if len(Loss_arr)>1:
        if (abs(Loss_arr[-2]-Loss_arr[-1]) < threshold):
              print("Loss delta: ", (Loss_arr[-2]-Loss_arr[-1]))
              break

    ################################################################

# HC:use pickle to transfer data log to analysis / plotting scripts

# summary file:
f_summary_handler = open('log_file_summary.pickle', 'wb')
summary_file = ['N:', input_N, '# of inner iterations:', len(Loss_arr)]
pickle.dump(summary_file, f_summary_handler, protocol=pickle.HIGHEST_PROTOCOL)

# Loss lists file:
f_loss_handler = open('data_log_loss.pickle', 'wb')
# insert N (size of hidden state) to the list as an indicator:
Loss_arr.insert(0,input_N)
pickle.dump(Loss_arr, f_loss_handler, protocol=pickle.HIGHEST_PROTOCOL)

# Out lists file (the data written out as a product of the NN):
f_out_handler = open('data_log_out.pickle', 'wb')
# insert N (size of hidden state) to the list as an indicator:
Out.insert(0, input_N)
pickle.dump(Out, f_out_handler, protocol=pickle.HIGHEST_PROTOCOL)

# Train perplexities file:
f__train_perplexity_handler = open('data_log_train_perp.pickle', 'wb')
# insert N (size of hidden state) to the list as an indicator:
training_perplexity.insert(0, input_N)
pickle.dump(training_perplexity, f__train_perplexity_handler, protocol=pickle.HIGHEST_PROTOCOL)

# Sample perplexities file:
f_sample_perplexity_handler = open('data_log_sample_perp.pickle', 'wb')
# insert N (size of hidden state) to the list as an indicator:
sample_perplexity.insert(0, input_N)
pickle.dump(sample_perplexity, f_sample_perplexity_handler, protocol=pickle.HIGHEST_PROTOCOL)

    ################################################################
