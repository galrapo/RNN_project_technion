import pickle

# HC:use pickle to receive data logs for plotting
#  TODO - define data structures, pass the data from pickle to the data structures and plot the data
# perplexities data structure (we are interested only in the last perplexity list element in each iteration


perplexity_sample = []
perplexity_loss = []


def summary_file():
    # summary file:
    f_summary_handler = open('log_file_summary.pickle', 'rb')
    summary_DB = pickle.load(f_summary_handler)
    for i in range(len(summary_DB)):
        print(summary_DB[i], end='  ')

def loss():
    f_loss_handler = open('data_log_loss.pickle', 'rb')
    loss_DB = pickle.load(f_loss_handler)
    print(loss_DB)

def out():
    # Out lists file (the data written out as a product of the NN):
    f_out_handler = open('data_log_out.pickle', 'rb')
    out_DB = pickle.load(f_out_handler)
    print(out_DB)

def train_perp():
    # Train perplexities file:
    f__train_perplexity_handler = open('data_log_train_perp.pickle', 'rb')
    train_perp_DB = pickle.load(f__train_perplexity_handler)
    print(train_perp_DB)
    # for i in range(len(train_perp_DB)):
    #     perplexity_loss.append(train_perp_DB[i][-1])
    # HC - plot the results:
    # plot_perplexity(perplexity_loss, N, "preplexity of loss func")

def sample_perp():
    # Sample perplexities file:
    f_sample_perplexity_handler = open('data_log_sample_perp.pickle', 'rb')
    sample_perp_DB = pickle.load(f_sample_perplexity_handler)
    print(sample_perp_DB)
    # for i in range(len(sample_perp_DB)):
    #     perplexity_sample.append(sample_perp_DB[i][-1])
    # # HC - plot the results:
    # plot_perplexity(perplexity_sample, N, "preplexity of sample func")

if __name__ == '__main__':
    summary_file()

