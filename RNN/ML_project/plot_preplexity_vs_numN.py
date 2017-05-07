import collections


import matplotlib.pyplot as plt
from script_for_graph import run_net_for_preplexity_calc
# from sklearn.model_selection import train_test_split


######       aux functions:         #######

#there you construct the unigram language model
##TODO - should it be erased??  -- we don't use it  ---  unigram model not needed
def unigram(tokens):
    model = collections.defaultdict(lambda: 0.01)
    for f in tokens:
        try:
            model [f] += 1
        except KeyError:
            model [f] = 1
            continue
    for word in model:
        model[word] = model[word]/float(len(model))
    return model


#computes perplexity of the unigram model on a testset
##TODO - should it be erased??  -- we don't use it
def perplexity(testset, model):
    testset = testset.split()
    perplexity = 1
    N = 0
    for word in testset:
        N += 1
        perplexity = perplexity * (1/model[word])
    perplexity = pow(perplexity, 1/float(N))
    return perplexity
#TODO - keep as is but do some adaptations

# plt.plot([1,2,3,4])
# plt.ylabel('some numbers')
# plt.show()
#
def plot_preplexity(preplexity , N, title):  #HC - modified 15.1.17
    '''
    graph preplexity over #N (preplexity is Y , #N is X)
    :param preplexity:  list of preplexities as calculated within the lossFunc
    for more info : look at https://en.wikipedia.org/wiki/Perplexity under Perplexity of a probability model
    :param N: list of N used ,  N is the net size (number of neurons)
    :return: None (plot is shown on the screen)
    '''
    plt.plot(N, preplexity, 'b-')  # args: x , y , type of plot
    plt.title("{}".format(title) + " (N)") # print title of each #HC - added 15.1.17
    plt.xlabel('N')#HC - added 15.1.17
    plt.ylabel('{}'.format(title))#HC - added 15.1.17
    # plt.axis([90,1000,0,100]) # axis ranges : X [0-100], Y [90-1000]
    plt.show()

# TODO - verify this -- new  from 9.1.17
def smaple_func_preplexity(loss,N): #HC - added 15.1.17
    '''
    calculating preplexity:
    for more info : look at https://en.wikipedia.org/wiki/Perplexity under Perplexity of a probability model
    :param loss: loss calculated
    :param targets: original y we should get as y_hat - for calculating loss func
    :return: preplexity value
    '''
    # loss = 0
    # for t in range(len(ps)):
    #     loss += -np.log(ps[t][targets[t], 0])  # softmax (cross-entropy loss)
    # N = len(ps) ## number of inputs / number of t's in ps (present state list) / number of iterations for calculating the loss function
    return  2**(loss/N) ## preplexity of a probability model


def run_preplexity_VS_N(): #HC - added 15.1.17
    '''
    calculated preplexity of loss function and sample function for a given N (size of net) - N list is predefined, will return preplexity for each N
    :return:  preplexity of loss fucniton list , preplexity of sample function list  -- for each N
    '''
    N = [25, 50, 100, 200, 400, 800] # predefined hidden layer sizes list - 100 was the original one
    i = 0
    preplexity_loss_func = []
    preplexity_sample_func = []
    for n in N:
        preplexity_loss_func[i], preplexity_sample_func[i] = run_net_for_preplexity_calc(n)
        i += 1
    return preplexity_loss_func, preplexity_sample_func, N


###############        main       ################
def main():
    # we first tokenize the text corpus
    corpus = open('shkpsr_RAW_TXT.txt', 'r').read()
    # tokens = nltk.word_tokenize(corpus)

    # model = unigram(tokens)
    # trainset, testset = train_test_split(corpus, train_size=0.8) #TODO - is it needed?
    # print(perplexity(trainset, model))
    # print(perplexity(testset, model))
    ##TODO - need to run iterations for each N (size of net) and calc prepxity, then plot it
    preplexity_loss, preplexity_sample, N_list = run_preplexity_VS_N()
    plot_preplexity(preplexity_loss, N_list, "preplexity of loss func")
    plot_preplexity(preplexity_sample, N_list, "preplexity of sample func")


###################################################



# main:
if __name__ == '__main__':
    main()