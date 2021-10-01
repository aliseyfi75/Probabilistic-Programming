import numpy as np
from scipy.special import loggamma

def joint_log_lik(doc_counts, topic_counts, alpha, gamma):
    """
    Calculate the joint log likelihood of the model
    
    Args:
        doc_counts: n_docs x n_topics array of counts per document of unique topics
        topic_counts: n_topics x alphabet_size array of counts per topic of unique words
        alpha: prior dirichlet parameter on document specific distributions over topics
        gamma: prior dirichlet parameter on topic specific distribuitons over words.
    Returns:
        ll: the joint log likelihood of the model
    """
    n_docs = doc_counts.shape[0]
    n_topics = doc_counts.shape[1]
    alphabet_size = topic_counts.shape[1]

    #From wikipedia:

    first_term = n_docs*(loggamma(n_topics * alpha)-n_topics*loggamma(alpha))
    posterier_doc_counts = doc_counts + alpha
    second_term = np.sum(np.sum(loggamma(posterier_doc_counts))) - np.sum(loggamma(np.sum(posterier_doc_counts, axis=1)))

    third_term = n_topics*(loggamma(alphabet_size*gamma) - alphabet_size*loggamma(gamma))
    posterier_topic_counts = topic_counts + gamma
    fourth_term = np.sum(np.sum(loggamma(posterier_topic_counts))) - np.sum(loggamma(np.sum(posterier_topic_counts, axis=1)))

    log_like = first_term + second_term + third_term + fourth_term

    return log_like
