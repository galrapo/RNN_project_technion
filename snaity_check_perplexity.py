# IMPORTS:
import matplotlib.pyplot as plt
import pickle
import subprocess
import sys

sanity_check = input('Enter type: full / test perp / loss:  ')

# HC - set the list of N - neural net size
# N = [25, 50, 100, 200, 400]
N = 100
loss_perplexity_full_DB = []


def plot_perplexity_for_specific_N(perplexity_list_for_N, n):
    '''
    sanity check to see if perplexity converges for a given N - plot the results
    :param perplexity_list_for_N:  list of perplexity for each iteration in the loss function
    '''
    x = range(len(perplexity_list_for_N))  # X axis parameter is the iteration # in lossFunc
    plt.plot(x, perplexity_list_for_N, 'r-')  # args: x , y , type of plot
    upper_title = 'Convergence of perplexity for the given N = ' + str(n)
    plt.suptitle("{}".format(upper_title), fontsize=16)  # print title of each
    last_iter_perp = perplexity_list_for_N[-1]
    lower_title = 'Perplexity at the last iteration:  ' + str(last_iter_perp)
    plt.title("{}".format(lower_title), fontsize=10)
    # plt.ylim([0, 20])
    plt.xlabel('Iteration #')
    plt.ylabel('{}'.format('Perplexity'))
    # plt.axis([90,1000,0,100]) # axis ranges : X [0-100], Y [90-1000]
    plt.show()


if sanity_check == 'full':
    # Train perplexities file:
    f__train_perplexity_handler_1 = open('pickle_files/pickle_25/data_log_train_perp.pickle', 'rb')
    f__train_perplexity_handler_2 = open('pickle_files/pickle_50/data_log_train_perp.pickle', 'rb')
    f__train_perplexity_handler_3 = open('pickle_files/pickle_100/data_log_train_perp.pickle', 'rb')
    f__train_perplexity_handler_4 = open('pickle_files/pickle_200/data_log_train_perp.pickle', 'rb')
    f__train_perplexity_handler_5 = open('pickle_files/pickle_400/data_log_train_perp.pickle', 'rb')
    # print('train_perplexity full:', file=backup)
    loss_perplexity_full_DB.append(pickle.load(f__train_perplexity_handler_1))
    loss_perplexity_full_DB.append(pickle.load(f__train_perplexity_handler_2))
    loss_perplexity_full_DB.append(pickle.load(f__train_perplexity_handler_3))
    loss_perplexity_full_DB.append(pickle.load(f__train_perplexity_handler_4))
    loss_perplexity_full_DB.append(pickle.load(f__train_perplexity_handler_5))
    for i in range(len(N)):
        # added a sanity function to check that perplexity for each N converges to optimum
        plot_perplexity_for_specific_N(loss_perplexity_full_DB[i], N[i])
    loss = []

    # Loss lists file:
    f_loss_handler_1 = open('pickle_files/pickle_25/data_log_loss.pickle', 'rb')
    f_loss_handler_2 = open('pickle_files/pickle_50/data_log_loss.pickle', 'rb')
    f_loss_handler_3 = open('pickle_files/pickle_100/data_log_loss.pickle', 'rb')
    f_loss_handler_4 = open('pickle_files/pickle_200/data_log_loss.pickle', 'rb')
    f_loss_handler_5 = open('pickle_files/pickle_400/data_log_loss.pickle', 'rb')
    # print('loss:', file=backup)
    loss.append(pickle.load(f_loss_handler_1))
    loss.append(pickle.load(f_loss_handler_2))
    loss.append(pickle.load(f_loss_handler_3))
    loss.append(pickle.load(f_loss_handler_4))
    loss.append(pickle.load(f_loss_handler_5))

    for i in range(len(N)):
        # added a sanity function to check that perplexity for each N converges to optimum
        plot_perplexity_for_specific_N(loss[i], N[i])

elif sanity_check == 'loss':
    f_loss_handler = open('data_log_loss.pickle', 'rb')
    Loss = pickle.load(f_loss_handler)
    x = range(len(Loss))  # X axis parameter is the iteration # in lossFunc
    plt.plot(x, Loss, 'r-')  # args: x , y , type of plot
    upper_title = 'Loss'
    plt.suptitle("{}".format(upper_title), fontsize=16)  # print title of each
    last_iter_perp = Loss[-1]
    lower_title = 'Loss at the last iteration:  ' + str(last_iter_perp)
    plt.title("{}".format(lower_title), fontsize=10)
    # plt.ylim([0, 20])
    plt.xlabel('Iteration #')
    plt.ylabel('{}'.format('Loss'))
    # plt.axis([90,1000,0,100]) # axis ranges : X [0-100], Y [90-1000]
    plt.show()
elif sanity_check == 'test perp':
    pass
else:
    pass