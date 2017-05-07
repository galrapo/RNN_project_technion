# IMPORTS:
import matplotlib.pyplot as plt
import pickle
import subprocess
import sys

backup = open('backup.txt', 'a')

# AUX FUNCTIONS:
def plot_perplexity(perplexity, N, title):
    '''
    graph perplexity over #N (perplexity is Y , #N is X)
    :param perplexity:  list of perplexities as calculated within the lossFunc
    for more info : look at https://en.wikipedia.org/wiki/Perplexity under Perplexity of a probability model
    :param N: list of N used ,  N is the net size (number of neurons)
    :return: None (plot is shown on the screen)
    '''
    plt.plot(N, perplexity, 'b-')  # args: x , y , type of plot
    plt.title("{}".format(title) + " (N)")  # print title of each
    plt.xlabel('N')
    plt.ylabel('{}'.format(title))
    # plt.axis([90,1000,0,100]) # axis ranges : X [0-100], Y [90-1000]
    plt.show()

# def parse_log_file_for_plot():#TODO - we probably don't need it and can use pickle instead.
#     '''
#     parse the log file to get preplexity and loss function of 'sample function' and 'loss function'
#     :return:
#     '''
#     pass

# HC - set the list of N - neural net size
N = [25, 50, 100, 200, 400]
# MAIN:
# HC - run the script 'script_for_project.py' for each size of N
# for n in N:
#     command_line = [sys.executable, "script_for_project.py", str(n)]
#     # process = subprocess.Popen(command_line)
#     # process.communicate()  # wait till the current process is done before resuming to the next iteration
#     # process.wait()
#     process = subprocess.call(command_line)

# HC:use pickle to receive data logs for plotting
# perplexities data structure (we are interested only in the last perplexity list element in each iteration
sapmle_perplexity_full_DB = []
loss_perplexity_full_DB = []
perplexity_sample = []
perplexity_loss = []
out = []
summary = []
loss = []

# summary files:
f_summary_handler_1 = open('pickle_files/pickle_25/log_file_summary.pickle', 'rb')
f_summary_handler_2 = open('pickle_files/pickle_50/log_file_summary.pickle', 'rb')
f_summary_handler_3 = open('pickle_files/pickle_100/log_file_summary.pickle', 'rb')
f_summary_handler_4 = open('pickle_files/pickle_200/log_file_summary.pickle', 'rb')
f_summary_handler_5 = open('pickle_files/pickle_400/log_file_summary.pickle', 'rb')
# print('Summary:', file=backup)
summary.append(pickle.load(f_summary_handler_1))
summary.append(pickle.load(f_summary_handler_2))
summary.append(pickle.load(f_summary_handler_3))
summary.append(pickle.load(f_summary_handler_4))
summary.append(pickle.load(f_summary_handler_5))
for i in range(len(summary)):
    for j in range(len(summary[i])):
        pass
        # print(summary[i][j], end='  ')
        # print(summary[i][j], end='  ', file=backup)
    # print()
    # print(file=backup)

# Loss lists file:
f_loss_handler_1 = open('pickle_files/pickle_25/data_log_loss.pickle', 'rb')
f_loss_handler_2 = open('pickle_files/pickle_50/data_log_loss.pickle', 'rb')
f_loss_handler_3 = open('pickle_files/pickle_100/data_log_loss.pickle', 'rb')
f_loss_handler_4 = open('pickle_files/pickle_200/data_log_loss.pickle', 'rb')
f_loss_handler_5 = open('pickle_files/pickle_400/data_log_loss.pickle', 'rb')
print('loss:', file=backup)
loss.append(pickle.load(f_loss_handler_1))
loss.append(pickle.load(f_loss_handler_2))
loss.append(pickle.load(f_loss_handler_3))
loss.append(pickle.load(f_loss_handler_4))
loss.append(pickle.load(f_loss_handler_5))
for i in range(len(loss)):
    # print(loss[i], file=backup)
    # print(loss[i])
    pass
# Out lists file (the data written out as a product of the NN):
f_out_handler_1 = open('pickle_files/pickle_25/data_log_out.pickle', 'rb')
f_out_handler_2 = open('pickle_files/pickle_50/data_log_out.pickle', 'rb')
f_out_handler_3 = open('pickle_files/pickle_100/data_log_out.pickle', 'rb')
f_out_handler_4 = open('pickle_files/pickle_200/data_log_out.pickle', 'rb')
f_out_handler_5 = open('pickle_files/pickle_400/data_log_out.pickle', 'rb')
# print('out:', file=backup)
out.append(pickle.load(f_out_handler_1))
out.append(pickle.load(f_out_handler_2))
out.append(pickle.load(f_out_handler_3))
out.append(pickle.load(f_out_handler_4))
out.append(pickle.load(f_out_handler_5))
for i in range(len(out)):
    # print(out[i], file=backup)
    # print(out[i])
    pass
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
    # print(loss_perplexity_full_DB[i], file=backup)
    perplexity_loss.append(loss_perplexity_full_DB[i][-1])
# HC - plot the results:
# print('perplexity loss:', file=backup)
# print(perplexity_loss, file=backup)
plot_perplexity(perplexity_loss, N, "perplexity of loss func")

# Sample perplexities file:
f_sample_perplexity_handler_1 = open('pickle_files/pickle_25/data_log_sample_perp.pickle', 'rb')
f_sample_perplexity_handler_2 = open('pickle_files/pickle_50/data_log_sample_perp.pickle', 'rb')
f_sample_perplexity_handler_3 = open('pickle_files/pickle_100/data_log_sample_perp.pickle', 'rb')
f_sample_perplexity_handler_4 = open('pickle_files/pickle_200/data_log_sample_perp.pickle', 'rb')
f_sample_perplexity_handler_5 = open('pickle_files/pickle_400/data_log_sample_perp.pickle', 'rb')
# print('sample perplexity DB full:', file=backup)
sapmle_perplexity_full_DB.append(pickle.load(f_sample_perplexity_handler_1))
sapmle_perplexity_full_DB.append(pickle.load(f_sample_perplexity_handler_2))
sapmle_perplexity_full_DB.append(pickle.load(f_sample_perplexity_handler_3))
sapmle_perplexity_full_DB.append(pickle.load(f_sample_perplexity_handler_4))
sapmle_perplexity_full_DB.append(pickle.load(f_sample_perplexity_handler_5))
for i in range(len(N)):
    # print(sapmle_perplexity_full_DB[i], file=backup)
    perplexity_sample.append(sapmle_perplexity_full_DB[i][-1])
# HC - plot the results:
# print('perplexity sample: ', file=backup)
# print(perplexity_sample, file=backup)
plot_perplexity(perplexity_sample, N, "perplexity of sample func")










