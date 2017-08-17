### IMPORTS:
import matplotlib.pyplot as plt
import pickle
import numpy as np

### AUX FUNCTION:

def lossPlotting(trainingLoss, testLoss, networkSize, optMethod):
    xArrayTestLoss = np.linspace(start=0, stop=len(trainingLoss), num=len(testLoss))
    xArrayTrainingLoss = range(len(trainingLoss))
    upperTitle = 'Training & Test Loss Values, Network Size = ' + str(networkSize) + ', OptMethod = ' + str(optMethod)
    plt.suptitle("{}".format(upperTitle), fontsize=14)
    lastIterValTest = testLoss[-1]
    lastIterValTraining = trainingLoss[-1]
    lowerTitle = 'Loss at last iteration - Training: ' + str(lastIterValTraining) + '  Test: ' + str(lastIterValTest)
    plt.title("{}".format(lowerTitle), fontsize=10)
    plt.xlabel('t')
    plt.ylabel('{}'.format('Loss'))
    plt.plot(xArrayTrainingLoss, trainingLoss, 'r-', label='Training Loss')
    plt.plot(xArrayTestLoss, testLoss, 'b-', label='Test Loss')
    plt.legend(loc='upper right')
    plt.show()

### PLOTTING FUNCTION DEFINITION:
## Description:
##      Plot data with desired x axis.
## Parameters:
##      y - list of data for Y axis
##      x - list of values for X axis
##      graphTitle - the title of the graph
##      titleY - the title for the y values
##      titleX - the title for the x values
## Returns:
##      None (plot is shown on screen)
def plotting(y, x, graphTitle, titleY, titleX):
    plt.plot(x, y, 'b-')                                                # Arguments: x vals , y vals , type of plot
    plt.suptitle("{}".format(graphTitle), fontsize=16)                  # print title of each
    lastIterVal = y[-1]
    plt.title("Last Iteration Value: " + " {}".format(lastIterVal), fontsize=10)
    plt.xlabel('{}'.format(titleX))
    plt.ylabel('{}'.format(titleY))
    plt.show()

##############################################################
####### USING PICKLE TO RECEIVE DATA LOGS FOR PLOTTING #######
##############################################################

N = [25, 50, 100, 200, 400]                 # RNN Size

# DATA STRUCTURES:
trainingLoss = []
trainingPerplexityFullDB = []
trainingPerplexityAdagrad = []
trainingPerplexityRMSprop = []
testLoss = []
testPerplexityFullDB = []
testPerplexityAdagrad = []
testPerplexityRMSprop = []
summary = []
out = []

# TRAINING LOSS LISTS FILES:
fTrainingLossHandler1 = open('pickle_25_newSet/TrainingLoss_ADAGRAD.pickle', 'rb')
fTrainingLossHandler2 = open('pickle_25_newSet/TrainingLoss_RMSPROP.pickle', 'rb')
fTrainingLossHandler3 = open('pickle_50_newSet/TrainingLoss_ADAGRAD.pickle', 'rb')
fTrainingLossHandler4 = open('pickle_50_newSet/TrainingLoss_RMSPROP.pickle', 'rb')
fTrainingLossHandler5 = open('pickle_100_newSet/TrainingLoss_ADAGRAD.pickle', 'rb')
fTrainingLossHandler6 = open('pickle_100_newSet/TrainingLoss_RMSPROP.pickle', 'rb')
fTrainingLossHandler7 = open('pickle_200_newSet/TrainingLoss_ADAGRAD.pickle', 'rb')
fTrainingLossHandler8 = open('pickle_200_newSet/TrainingLoss_RMSPROP.pickle', 'rb')
fTrainingLossHandler9 = open('pickle_400_newSet/TrainingLoss_ADAGRAD.pickle', 'rb')
fTrainingLossHandler10 = open('pickle_400_newSet/TrainingLoss_RMSPROP.pickle', 'rb')
trainingLoss.append(pickle.load(fTrainingLossHandler1))
trainingLoss.append(pickle.load(fTrainingLossHandler2))
trainingLoss.append(pickle.load(fTrainingLossHandler3))
trainingLoss.append(pickle.load(fTrainingLossHandler4))
trainingLoss.append(pickle.load(fTrainingLossHandler5))
trainingLoss.append(pickle.load(fTrainingLossHandler6))
trainingLoss.append(pickle.load(fTrainingLossHandler7))
trainingLoss.append(pickle.load(fTrainingLossHandler8))
trainingLoss.append(pickle.load(fTrainingLossHandler9))
trainingLoss.append(pickle.load(fTrainingLossHandler10))

plotting(trainingLoss[0], range(len(trainingLoss[0])), 'ADAGRAD Training Loss (N=25)', 'Loss', 't')
plotting(trainingLoss[1], range(len(trainingLoss[1])), 'RMSPROP Training Loss (N=25)', 'Loss', 't')
plotting(trainingLoss[2], range(len(trainingLoss[2])), 'ADAGRAD Training Loss (N=50)', 'Loss', 't')
plotting(trainingLoss[3], range(len(trainingLoss[3])), 'RMSPROP Training Loss (N=50)', 'Loss', 't')
plotting(trainingLoss[4], range(len(trainingLoss[4])), 'ADAGRAD Training Loss (N=100)', 'Loss', 't')
plotting(trainingLoss[5], range(len(trainingLoss[5])), 'RMSPROP Training Loss (N=100)', 'Loss', 't')
plotting(trainingLoss[6], range(len(trainingLoss[6])), 'ADAGRAD Training Loss (N=200)', 'Loss', 't')
plotting(trainingLoss[7], range(len(trainingLoss[7])), 'RMSPROP Training Loss (N=200)', 'Loss', 't')
plotting(trainingLoss[8], range(len(trainingLoss[8])), 'ADAGRAD Training Loss (N=400)', 'Loss', 't')
plotting(trainingLoss[9], range(len(trainingLoss[9])), 'RMSPROP Training Loss (N=400)', 'Loss', 't')

# TEST LOSS LISTS FILES:
fTestLossHandler1 = open('pickle_25_newSet/TestLoss_ADAGRAD.pickle', 'rb')
fTestLossHandler2 = open('pickle_25_newSet/TestLoss_RMSPROP.pickle', 'rb')
fTestLossHandler3 = open('pickle_50_newSet/TestLoss_ADAGRAD.pickle', 'rb')
fTestLossHandler4 = open('pickle_50_newSet/TestLoss_RMSPROP.pickle', 'rb')
fTestLossHandler5 = open('pickle_100_newSet/TestLoss_ADAGRAD.pickle', 'rb')
fTestLossHandler6 = open('pickle_100_newSet/TestLoss_RMSPROP.pickle', 'rb')
fTestLossHandler7 = open('pickle_200_newSet/TestLoss_ADAGRAD.pickle', 'rb')
fTestLossHandler8 = open('pickle_200_newSet/TestLoss_RMSPROP.pickle', 'rb')
fTestLossHandler9 = open('pickle_400_newSet/TestLoss_ADAGRAD.pickle', 'rb')
fTestLossHandler10 = open('pickle_400_newSet/TestLoss_RMSPROP.pickle', 'rb')
testLoss.append(pickle.load(fTestLossHandler1))
testLoss.append(pickle.load(fTestLossHandler2))
testLoss.append(pickle.load(fTestLossHandler3))
testLoss.append(pickle.load(fTestLossHandler4))
testLoss.append(pickle.load(fTestLossHandler5))
testLoss.append(pickle.load(fTestLossHandler6))
testLoss.append(pickle.load(fTestLossHandler7))
testLoss.append(pickle.load(fTestLossHandler8))
testLoss.append(pickle.load(fTestLossHandler9))
testLoss.append(pickle.load(fTestLossHandler10))

plotting(testLoss[0], range(len(testLoss[0])), 'ADAGRAD Test Loss (N=25)', 'Loss', 't')
plotting(testLoss[1], range(len(testLoss[1])), 'RMSPROP Test Loss (N=25)', 'Loss', 't')
plotting(testLoss[2], range(len(testLoss[2])), 'ADAGRAD Test Loss (N=50)', 'Loss', 't')
plotting(testLoss[3], range(len(testLoss[3])), 'RMSPROP Test Loss (N=50)', 'Loss', 't')
plotting(testLoss[4], range(len(testLoss[4])), 'ADAGRAD Test Loss (N=100)', 'Loss', 't')
plotting(testLoss[5], range(len(testLoss[5])), 'RMSPROP Test Loss (N=100)', 'Loss', 't')
plotting(testLoss[6], range(len(testLoss[6])), 'ADAGRAD Test Loss (N=200)', 'Loss', 't')
plotting(testLoss[7], range(len(testLoss[7])), 'RMSPROP Test Loss (N=200)', 'Loss', 't')
plotting(testLoss[8], range(len(testLoss[8])), 'ADAGRAD Test Loss (N=400)', 'Loss', 't')
plotting(testLoss[9], range(len(testLoss[9])), 'RMSPROP Test Loss (N=400)', 'Loss', 't')

lossPlotting(trainingLoss[0], testLoss[0], 25, "ADAGRAD")
lossPlotting(trainingLoss[1], testLoss[1], 25, "RMSPROP")
lossPlotting(trainingLoss[2], testLoss[2], 50, "ADAGRAD")
lossPlotting(trainingLoss[3], testLoss[3], 50, "RMSPROP")
lossPlotting(trainingLoss[4], testLoss[4], 100, "ADAGRAD")
lossPlotting(trainingLoss[5], testLoss[5], 100, "RMSPROP")
lossPlotting(trainingLoss[6], testLoss[6], 200, "ADAGRAD")
lossPlotting(trainingLoss[7], testLoss[7], 200, "RMSPROP")
lossPlotting(trainingLoss[8], testLoss[8], 400, "ADAGRAD")
lossPlotting(trainingLoss[9], testLoss[9], 400, "RMSPROP")

# TRAINING PERPLEXITY FILES:
fTrainingPerplexityHandler1 = open('pickle_25_newSet/TrainingPerplexity_ADAGRAD.pickle', 'rb')
fTrainingPerplexityHandler2 = open('pickle_25_newSet/TrainingPerplexity_RMSPROP.pickle', 'rb')
fTrainingPerplexityHandler3 = open('pickle_50_newSet/TrainingPerplexity_ADAGRAD.pickle', 'rb')
fTrainingPerplexityHandler4 = open('pickle_50_newSet/TrainingPerplexity_RMSPROP.pickle', 'rb')
fTrainingPerplexityHandler5 = open('pickle_100_newSet/TrainingPerplexity_ADAGRAD.pickle', 'rb')
fTrainingPerplexityHandler6 = open('pickle_100_newSet/TrainingPerplexity_RMSPROP.pickle', 'rb')
fTrainingPerplexityHandler7 = open('pickle_200_newSet/TrainingPerplexity_ADAGRAD.pickle', 'rb')
fTrainingPerplexityHandler8 = open('pickle_200_newSet/TrainingPerplexity_RMSPROP.pickle', 'rb')
fTrainingPerplexityHandler9 = open('pickle_400_newSet/TrainingPerplexity_ADAGRAD.pickle', 'rb')
fTrainingPerplexityHandler10 = open('pickle_400_newSet/TrainingPerplexity_RMSPROP.pickle', 'rb')
trainingPerplexityFullDB.append(pickle.load(fTrainingPerplexityHandler1))
trainingPerplexityFullDB.append(pickle.load(fTrainingPerplexityHandler2))
trainingPerplexityFullDB.append(pickle.load(fTrainingPerplexityHandler3))
trainingPerplexityFullDB.append(pickle.load(fTrainingPerplexityHandler4))
trainingPerplexityFullDB.append(pickle.load(fTrainingPerplexityHandler5))
trainingPerplexityFullDB.append(pickle.load(fTrainingPerplexityHandler6))
trainingPerplexityFullDB.append(pickle.load(fTrainingPerplexityHandler7))
trainingPerplexityFullDB.append(pickle.load(fTrainingPerplexityHandler8))
trainingPerplexityFullDB.append(pickle.load(fTrainingPerplexityHandler9))
trainingPerplexityFullDB.append(pickle.load(fTrainingPerplexityHandler10))

for i in range(len(N)):
    trainingPerplexityAdagrad.append(trainingPerplexityFullDB[2*i][-1])
    trainingPerplexityRMSprop.append(trainingPerplexityFullDB[2*i+1][-1])

# Adagrad Perplexity plot:
plotting(trainingPerplexityAdagrad, N, "ADAGRAD Training Perplexity", 'Perplexity', 'N')        # Plot the results
# RMSprop Perplexity plot:
plotting(trainingPerplexityRMSprop, N, "RMSPROP Training Perplexity", 'Perplexity', 'N')        # Plot the results

# TEST PERPLEXITY FILES:
fTestPerplexityHandler1 = open('pickle_25_newSet/TestPerplexity_ADAGRAD.pickle', 'rb')
fTestPerplexityHandler2 = open('pickle_25_newSet/TestPerplexity_RMSPROP.pickle', 'rb')
fTestPerplexityHandler3 = open('pickle_50_newSet/TestPerplexity_ADAGRAD.pickle', 'rb')
fTestPerplexityHandler4 = open('pickle_50_newSet/TestPerplexity_RMSPROP.pickle', 'rb')
fTestPerplexityHandler5 = open('pickle_100_newSet/TestPerplexity_ADAGRAD.pickle', 'rb')
fTestPerplexityHandler6 = open('pickle_100_newSet/TestPerplexity_RMSPROP.pickle', 'rb')
fTestPerplexityHandler7 = open('pickle_200_newSet/TestPerplexity_ADAGRAD.pickle', 'rb')
fTestPerplexityHandler8 = open('pickle_200_newSet/TestPerplexity_RMSPROP.pickle', 'rb')
fTestPerplexityHandler9 = open('pickle_400_newSet/TestPerplexity_ADAGRAD.pickle', 'rb')
fTestPerplexityHandler10 = open('pickle_400_newSet/TestPerplexity_RMSPROP.pickle', 'rb')
testPerplexityFullDB.append(pickle.load(fTestPerplexityHandler1))
testPerplexityFullDB.append(pickle.load(fTestPerplexityHandler2))
testPerplexityFullDB.append(pickle.load(fTestPerplexityHandler3))
testPerplexityFullDB.append(pickle.load(fTestPerplexityHandler4))
testPerplexityFullDB.append(pickle.load(fTestPerplexityHandler5))
testPerplexityFullDB.append(pickle.load(fTestPerplexityHandler6))
testPerplexityFullDB.append(pickle.load(fTestPerplexityHandler7))
testPerplexityFullDB.append(pickle.load(fTestPerplexityHandler8))
testPerplexityFullDB.append(pickle.load(fTestPerplexityHandler9))
testPerplexityFullDB.append(pickle.load(fTestPerplexityHandler10))

for i in range(len(N)):
    testPerplexityAdagrad.append(testPerplexityFullDB[2*i][-1])
    testPerplexityRMSprop.append(testPerplexityFullDB[2*i+1][-1])

# Adagrad Perplexity plot:
plotting(testPerplexityAdagrad, N, "ADAGRAD Test Perplexity", 'Perplexity', 'N')        # Plot the results
# RMSprop Perplexity plot:
plotting(testPerplexityRMSprop, N, "RMSPROP Test Perplexity", 'Perplexity', 'N')        # Plot the results

# SUMMARY FILES:
fSummaryHandler1 = open('pickle_25/Summary_ADAGRAD.pickle', 'rb')
fSummaryHandler2 = open('pickle_25/Summary_RMSPROP.pickle', 'rb')
fSummaryHandler3 = open('pickle_50/Summary_ADAGRAD.pickle', 'rb')
fSummaryHandler4 = open('pickle_50/Summary_RMSPROP.pickle', 'rb')
fSummaryHandler5 = open('pickle_100/Summary_ADAGRAD.pickle', 'rb')
fSummaryHandler6 = open('pickle_100/Summary_RMSPROP.pickle', 'rb')
fSummaryHandler7 = open('pickle_200/Summary_ADAGRAD.pickle', 'rb')
fSummaryHandler8 = open('pickle_200/Summary_RMSPROP.pickle', 'rb')
fSummaryHandler9 = open('pickle_400/Summary_ADAGRAD.pickle', 'rb')
fSummaryHandler10 = open('pickle_400/Summary_RMSPROP.pickle', 'rb')
summary.append(pickle.load(fSummaryHandler1))
summary.append(pickle.load(fSummaryHandler2))
summary.append(pickle.load(fSummaryHandler3))
summary.append(pickle.load(fSummaryHandler4))
summary.append(pickle.load(fSummaryHandler5))
summary.append(pickle.load(fSummaryHandler6))
summary.append(pickle.load(fSummaryHandler7))
summary.append(pickle.load(fSummaryHandler8))
summary.append(pickle.load(fSummaryHandler9))
summary.append(pickle.load(fSummaryHandler10))

# OUT LISTS FILES (THE DATA WRITTEN OUT AS A PRODUCT OF THE RNN):
fOutHandler1 = open('pickle_25/Out_ADAGRAD.pickle', 'rb')
fOutHandler2 = open('pickle_25/Out_RMSPROP.pickle', 'rb')
fOutHandler3 = open('pickle_50/Out_ADAGRAD.pickle', 'rb')
fOutHandler4 = open('pickle_50/Out_RMSPROP.pickle', 'rb')
fOutHandler5 = open('pickle_100/Out_ADAGRAD.pickle', 'rb')
fOutHandler6 = open('pickle_100/Out_RMSPROP.pickle', 'rb')
fOutHandler7 = open('pickle_200/Out_ADAGRAD.pickle', 'rb')
fOutHandler8 = open('pickle_200/Out_RMSPROP.pickle', 'rb')
fOutHandler9 = open('pickle_400/Out_ADAGRAD.pickle', 'rb')
fOutHandler10 = open('pickle_400/Out_RMSPROP.pickle', 'rb')
out.append(pickle.load(fOutHandler1))
out.append(pickle.load(fOutHandler2))
out.append(pickle.load(fOutHandler3))
out.append(pickle.load(fOutHandler4))
out.append(pickle.load(fOutHandler5))
out.append(pickle.load(fOutHandler6))
out.append(pickle.load(fOutHandler7))
out.append(pickle.load(fOutHandler8))
out.append(pickle.load(fOutHandler9))
out.append(pickle.load(fOutHandler10))