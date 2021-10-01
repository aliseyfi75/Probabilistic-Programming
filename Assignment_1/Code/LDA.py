from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

from joint_log_lik import joint_log_lik
from sample_topic_assignment import sample_topic_assignment

import wandb

wandb.init(project='LDA', entity='aliseyfi')

bagofwords = loadmat('Data/bagofwords_nips.mat')
WS = bagofwords['WS'][0] - 1  #go to 0 indexed
DS = bagofwords['DS'][0] - 1

WO = loadmat('Data/words_nips.mat')['WO'][:,0]
titles = loadmat('Data/titles_nips.mat')['titles'][:,0]


#This script outlines how you might create a MCMC sampler for the LDA model

alphabet_size = WO.size

document_assignment = DS
words = WS

#subset data, EDIT THIS PART ONCE YOU ARE CONFIDENT THE MODEL IS WORKING
#PROPERLY IN ORDER TO USE THE ENTIRE DATA SET
# words = words[document_assignment < 100]
# document_assignment  = document_assignment[document_assignment < 100]

n_docs = document_assignment.max() + 1

#number of topics
n_topics = 20

#initial topic assigments
topic_assignment = np.random.randint(n_topics, size=document_assignment.size)

#within document count of topics
doc_counts = np.zeros((n_docs,n_topics))

for d in range(n_docs):
    #histogram counts the number of occurences in a certain defined bin
    doc_counts[d] = np.histogram(topic_assignment[document_assignment == d], bins=n_topics, range=(-0.5,n_topics-0.5))[0]

#doc_N: array of size n_docs count of total words in each document, minus 1
doc_N = doc_counts.sum(axis=1) - 1

#within topic count of words
topic_counts = np.zeros((n_topics,alphabet_size))

for k in range(n_topics):
    w_k = words[topic_assignment == k]

    topic_counts[k] = np.histogram(w_k, bins=alphabet_size, range=(-0.5,alphabet_size-0.5))[0]



#topic_N: array of size n_topics count of total words assigned to each topic
topic_N = topic_counts.sum(axis=1)

#prior parameters, alpha parameterizes the dirichlet to regularize the
#document specific distributions over topics and gamma parameterizes the 
#dirichlet to regularize the topic specific distributions over words.
#These parameters are both scalars and really we use alpha * ones() to
#parameterize each dirichlet distribution. Iters will set the number of
#times your sampler will iterate.
alpha = 1/n_topics
gamma = 1/n_topics
iters = 1000


jll = []
for i in range(iters):
    jll.append(joint_log_lik(doc_counts,topic_counts,alpha,gamma))

    prm = np.random.permutation(words.shape[0])

    words = words[prm]
    document_assignment = document_assignment[prm]
    topic_assignment = topic_assignment[prm]

    topic_assignment, topic_counts, doc_counts, topic_N = sample_topic_assignment(
                                topic_assignment,
                                topic_counts,
                                doc_counts,
                                topic_N,
                                doc_N,
                                alpha,
                                gamma,
                                words,
                                document_assignment)
    wandb.log({"joint log likelihood": jll[-1]})
jll.append(joint_log_lik(doc_counts,topic_counts,alpha,gamma))


### find the 10 most probable words of the 20 topics:
fstr = ''

for t in range(n_topics):
    most_probable = topic_counts[t,:].argsort()[-10:]
    for i in range(10):
        fstr = fstr + str(WO[most_probable[9-i]][0]) + ', '
    fstr += '\n'

with open('most_probable_words_per_topic','w') as f:
    f.write(fstr)


#most similar documents to document 0 by cosine similarity over topic distribution:
#normalize topics per document and dot product:

doc_counts_row_sums = doc_counts.sum(axis=1)
normalized_doc_counts = doc_counts/doc_counts_row_sums[:,np.newaxis]

most_similar = 0
score = np.ones(normalized_doc_counts.shape[0])

for d in range(1,doc_counts.shape[0]):
    score[d] = np.dot(normalized_doc_counts[0,:], normalized_doc_counts[d,:])

best_scores = score.argsort()[-10:]
fstr = ''
for i in range(2,11):
    fstr += str(best_scores[10-i]) + ' with score of: ' + str(score[best_scores[10-i]]) +'\n'

with open('most_similar_titles_to_0','w') as f:
    f.write(fstr)
