### IMPORTS:
import numpy as np
import pickle
import os
import math
import sys

### PREPARE INPUT & OUTPUT FOLDERS:
sizeN = int(input('Insert N (hidden layer size):  '))       # Summon any run of the script with different N
logFile = open('LogFile.txt', 'a')
myPath = os.path.dirname(os.path.realpath(__file__))        # Get current directory
newPath = r"pickle_" + str(sizeN) + "_newSet"               # Set new folder name
concatPath = myPath + "/" + newPath                         # Set path of new folder
if not os.path.exists(concatPath):                          # Create new folder if it doesnt exit
    os.makedirs(concatPath)

# PREPARE FULL DATA - TRAINING SET + TEST SET MERGED:
filenames = ['ShakespeareTrainingSetNew2.txt', 'ShakespeareTestSetNew2.txt']
with open('MergedDataSetNew2.txt', 'w') as outputFile:
    for f in filenames:
        with open(f) as inputFile:
            for line in inputFile:
                outputFile.write(line)
mergedData = open('MergedDataSetNew2.txt', 'r').read()      # File that includes both training set & test set data

### PREPARE TRAINING SET:
data = open('ShakespeareTrainingSetNew2.txt', 'r').read()   # Should be a simple plain text file
dataSize = len(data)
vocab = list(set(mergedData))                               # Vocabulary
vocabSize = len(vocab)
charToIndex = { ch:i for i,ch in enumerate(vocab) }         # Vocabulary to index value - get index by writing charToIndex['desiredChar']
indexToChar = { i:ch for i,ch in enumerate(vocab) }         # Index value to vocabulary - get index by writing indexToChar['desiredIndex']

### PREPARE TEST SET:
testData = open('ShakespeareTestSetNew2.txt', 'r').read()   # Should be a simple plain text file
testDataSize = len(testData)
# testVocab = list(set(mergedData))                         # Vocabulary
# testVocabSize = len(testVocab)

### HYPERPARAMETERS:
hiddenSize = sizeN                                          # Size of hidden layer of neurons - given as input from cmd line
seqLength = 25                                              # Number of steps to unroll the RNN for
threshold = 0.0001                                          # Treshold size for gradient descent - will stop optimization when reaching a change <= given threshold
optimizationMethod = "ADAGRAD"                              # Set to one of the following: ADAGRAD / RMSPROP
gamma = 0.9                                                 # The gamma parameter for RMSProp optimization
if optimizationMethod == "ADAGRAD":
    learningRate = 1e-1                                     # The size of the step for gradient descent
if optimizationMethod == "RMSPROP":
    learningRate = 1e-3                                     # The size of the step for gradient descent

### MODEL PARAMETERS:
Wxh = np.random.randn(hiddenSize, vocabSize)*0.01           # Input to hidden matrix
Whh = np.random.randn(hiddenSize, hiddenSize)*0.01          # Hidden to hidden matrix
Why = np.random.randn(vocabSize, hiddenSize)*0.01           # Hidden to output matrix
bh = np.zeros((hiddenSize, 1))                              # Hidden bias vector
by = np.zeros((vocabSize, 1))                               # Output bias vector


### LOSS FUNCTION DEFINITION:
## Parameters:
##      inputs - a list of integers, each integer representing an index of a vocabulary char
##      targets - a list of integers, each integer representing an index of the target vocabulary char
##      hprev - Hx1 array holding the initial hidden state
## Returns:
##      (1) Loss value
##      (2) Gradients on model parameters
##      (3) Last hidden state
##      (4) Y values
##      (5) Perplexity value
def lossFun(inputs, targets, hprev):
    xs, hs, ys, ps = {}, {}, {}, {}
    loss = 0
    counter = 0
    hs[-1] = np.copy(hprev)                                             # Assigning hprev to last column of hs for the upcoming loop

    # FORWARD PASS:
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocabSize,1))                                 # Encode input in a 1-of-k representation - set vector
        xs[t][inputs[t]] = 1                                            # Encode input in a 1-of-k representation - write '1' in correct index
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # Hidden state calculation
        ys[t] = np.dot(Why, hs[t]) + by                                 # Un-normalized log probabilities for next char
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))                   # Softmax function to determine probability vector for next char
        loss += -np.log(ps[t][targets[t],0])                            # Softmax - cross-entropy loss
        counter += 1                                                    # Incrementing counter for future perplexity calculation

    loss = loss / seqLength                                             # Normalize calculated result by sequence length
    perplexity = 2 ** (loss)                                            # Perplexity calculation - ask Sasha if needs another division

    # BACKWARD PASS - COMPUTE GRADIENTS GOING BACKWARDS:
    # http://cs231n.github.io/neural-networks-case-study/#grad
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])
    for t in reversed(range(len(inputs))):
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1
        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        dh = np.dot(Why.T, dy) + dhnext
        dhraw = (1 - hs[t] * hs[t]) * dh
        dbh += dhraw
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t-1].T)
        dhnext = np.dot(Whh.T, dhraw)
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam)                              # Clip gradients to mitigate exploding gradients

    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1], ys, perplexity

### SAMPLE FUNCTION DEFINITION:
## Description:
##      Sample the test set & produce output, along with loss & perplexity calculations
## Parameters:
##      hprev - the memory state
##      sampleData - the complete test set data of the network
## Returns:
##      (1) An indices vector, representing the sampling output text by indices
##      (2) The sample function perplexity
##      (3) The sample function loss
def sampleFun(sampleData, hprev):

    inputs = sampleData[:-1]
    targets = sampleData[1:]

    xs, hs, ys, ps = {}, {}, {}, {}
    loss = 0
    indices = []
    hs[-1] = np.copy(hprev)                                             # Assigning hprev to last column of hs for the upcoming loop

    # FORWARD PASS:
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocabSize,1))                                 # Encode input in a 1-of-k representation - set vector
        xs[t][inputs[t]] = 1                                            # Encode input in a 1-of-k representation - write '1' in correct index
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # Hidden state calculation
        ys[t] = np.dot(Why, hs[t]) + by                                 # Un-normalized log probabilities for next char
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))                   # Softmax function to determine probability vector for next char
        loss += -np.log(ps[t][targets[t],0])                            # Softmax - cross-entropy loss
        index = np.random.choice(range(vocabSize), p=ps[t].ravel())     # Choosing a char inde randomly, based on the probability distribution p
        indices.append(index)

    loss = loss / len(inputs)                                           # Normalize calculated result by testDataSize
    perplexity = 2 ** (loss)                                            # Perplexity calculation

    return indices, perplexity, loss

### INITIALIZATION:
n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)   # Memory variables for optimization
mbh, mby = np.zeros_like(bh), np.zeros_like(by)                                 # Memory variables for optimization
smoothLoss = -np.log2(1.0/vocabSize)*seqLength                                  # Loss at iteration #0
trainingLoss = []
trainingPerplexity = []
testLoss = []
testPerplexity = []
Out = []
txt = ''

while True:

    # PREPARE INPUTS (SWEEPING FROM LEFT TO RIGHT IN seqLength STEPS LONG:
    if  n == 0:
        hprev = np.zeros((hiddenSize,1))                                        # Reset RNN memory
        p = 0

    # SAMPLE FROM THE MODEL (each time an iteration over the whole text finishes):                  # HC - we changed the sample function to perform an open circuit
    if p + seqLength + 1 >= len(data):                                                              # HC - modified:  n % 100 == 0  --> p+seq_length+1 >= len(data)
        sampleData = [charToIndex[ch] for ch in testData]                                           # HC - the whole text in characters
        testIndices, testPerplexityIteration, testLossIteration = sampleFun(sampleData, hprev)      # HC
        txt = ''.join(indexToChar[ix] for ix in testIndices)                                        # HC
        testPerplexity.append(testPerplexityIteration)                                              # HC
        testLoss.append(testLossIteration)                                                          # HC

        # PREPARE INPUTS (SWEEPING FROM LEFT TO RIGHT IN seqLength STEPS LONG:
        hprev = np.zeros((hiddenSize, 1))                                       # Reset RNN memory
        p = 0                                                                   # Begin at start of data

    inputs = [charToIndex[ch] for ch in data[p:p+seqLength]]                    # Set inputs as indices corresponding to appropriate data chars range
    targets = [charToIndex[ch] for ch in data[p+1:p+seqLength+1]]               # Set targets as indices corresponding to appropriate data chars range
                                                    
    # FORWARD seqLength CHARS THROUGH THE NET & FETCH GRADIENT:
    loss, dWxh, dWhh, dWhy, dbh, dby, hprev, y, trainingPerplexityIteration = lossFun(inputs, targets, hprev)
    trainingPerplexity.append(trainingPerplexityIteration)
    smoothLoss = smoothLoss * 0.999 + loss * 0.001

    if optimizationMethod == "ADAGRAD":
        # ADAGRAD PARAMETER UPDATE:
        for parameter, dparameter, mem in zip([Wxh, Whh, Why, bh, by],
                                              [dWxh, dWhh, dWhy, dbh, dby],
                                              [mWxh, mWhh, mWhy, mbh, mby]):
            mem += dparameter * dparameter                                      # Add the square of the calculated gradient to the sum stored in 'mem'
            parameter += -learningRate * dparameter / np.sqrt(mem + 1e-8)       # Adagrad update
    elif optimizationMethod == "RMSPROP":
        # RMSPROP PARAMETER UPDATE:
        for parameter, dparameter, mem in zip([Wxh, Whh, Why, bh, by],
                                                           [dWxh, dWhh, dWhy, dbh, dby],
                                                           [mWxh, mWhh, mWhy, mbh, mby]):
            mem = (gamma * mem) + ((1 - gamma) * (dparameter * dparameter))     # Add the square of the calculated gradient to the sum stored in 'mem'
            parameter += -learningRate * dparameter / np.sqrt(mem + 1e-8)       # RMSProp update

    p += seqLength                                                              # Move data pointer by 'seqLength'
    n += 1                                                                      # Increment iteration counter
    trainingLoss.append(smoothLoss)
    Out.append(txt)

    # STOP CONDITION - CHECK IF DELTA OF THE LOSS BETWEEN ITERATIONS IS SMALLER THAN DESIRED THRESHOLD:
    if len(trainingLoss)>1:
        if (abs(trainingLoss[-2]-trainingLoss[-1]) < threshold):
              print("Loss delta: ",(trainingLoss[-2]-trainingLoss[-1]))
              break


##############################################################################
####### USING PICKLE TO TRANSFER DATA LOG TO ANALYSIS/PLOTTING SCRIPTS #######
##############################################################################

# TRAINING LOSS LISTS FILE:
fTrainingLossHandler = open("pickle_" + str(sizeN) + "_newSet/TrainingLoss_" + optimizationMethod + ".pickle", 'wb')
trainingLoss.insert(0,sizeN)
pickle.dump(trainingLoss, fTrainingLossHandler, protocol=pickle.HIGHEST_PROTOCOL)

# TEST LOSS LISTS FILE:
fTestLossHandler = open("pickle_" + str(sizeN) + "_newSet/TestLoss_" + optimizationMethod + ".pickle", 'wb')
testLoss.insert(0,sizeN)
pickle.dump(testLoss, fTestLossHandler, protocol=pickle.HIGHEST_PROTOCOL)

# TRAINING PERPLEXITIES FILE:
fTrainPerplexityHandler = open("pickle_" + str(sizeN) + "_newSet/TrainingPerplexity_" + optimizationMethod + ".pickle", 'wb')
trainingPerplexity.insert(0, sizeN)
pickle.dump(trainingPerplexity, fTrainPerplexityHandler, protocol=pickle.HIGHEST_PROTOCOL)

# TEST PERPLEXITIES FILE:
fSamplePerplexityHandler = open("pickle_" + str(sizeN) + "_newSet/TestPerplexity_" + optimizationMethod + ".pickle", 'wb')
testPerplexity.insert(0, sizeN)
pickle.dump(testPerplexity, fSamplePerplexityHandler, protocol=pickle.HIGHEST_PROTOCOL)

# SUMMARY FILE:
fSummaryHandler = open("pickle_" + str(sizeN) + "_newSet/Summary_" + optimizationMethod + ".pickle", 'wb')
summaryFile = ['N:', sizeN, '# of inner iterations:', len(trainingLoss)]
pickle.dump(summaryFile, fSummaryHandler, protocol=pickle.HIGHEST_PROTOCOL)

# OUT LISTS FILE (THE DATA WRITTEN OUT AS A PRODUCT OF THE RNN):
fOutHandler = open("pickle_" + str(sizeN) + "_newSet/Out_" + optimizationMethod + ".pickle", 'wb')
Out.insert(0,sizeN)
pickle.dump(Out, fOutHandler, protocol=pickle.HIGHEST_PROTOCOL)

##############################################################################
##############################################################################
##############################################################################

### OLD SAMPLE FUNCTION DEFINITION:
## Description:
##      Sample a sequence of integers from the model
## Parameters:
##      h - the memory state
##      seedIndex - the seed letter for first time step
## Returns:
##      (1) An indices vector, representing the sampling output text by indices
##      (2) The sample function perplexity
# def sampleFun(h, seedIndex):
#     x = np.zeros((vocabSize, 1))
#     x[seedIndex] = 1
#     indices = []
#     p = []
#
#     for t in range(200):                                            # need to change to:  for t in range(len(seedIndex))
#         h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
#         y = np.dot(Why, h) + by
#         p = np.exp(y) / np.sum(np.exp(y))                           # Softmax function to determine probability vector for next char
#         ix = np.random.choice(range(vocabSize), p=p.ravel())        # Choosing a char inde randomly, based on the probability distribution p
#         x = np.zeros((vocabSize, 1))
#         x[ix] = 1
#         indices.append(ix)
#
#     # SAMPLE FUNCTION PERPLEXITY CALCULATION:
#     auxPerp = float(0)
#     for i in range(len(p)):
#         auxPerp += -np.log2(float(p[i]))
#     samPerplexity = 2**(auxPerp / len(p))
#
#     return indices, samPerplexity