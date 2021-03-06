import numpy as np

def sample_topic_assignment(topic_assignment,
                            topic_counts,
                            doc_counts,
                            topic_N,
                            doc_N,
                            alpha,
                            gamma,
                            words,
                            document_assignment):
    """
    Sample the topic assignment for each word in the corpus, one at a time.
    
    Args:
        topic_assignment: size n array of topic assignments
        topic_counts: n_topics x alphabet_size array of counts per topic of unique words        
        doc_counts: n_docs x n_topics array of counts per document of unique topics

        topic_N: array of size n_topics count of total words assigned to each topic
        doc_N: array of size n_docs count of total words in each document, minus 1
        
        alpha: prior dirichlet parameter on document specific distributions over topics
        gamma: prior dirichlet parameter on topic specific distribuitons over words.

        words: size n array of wors
        document_assignment: size n array of assignments of words to documents
    Returns:
        topic_assignment: updated topic_assignment array
        topic_counts: updated topic counts array
        doc_counts: updated doc_counts array
        topic_N: updated count of words assigned to each topic
    """
    
    # I get helped from this article to understand the concept and implement this part:
    # https://www.ics.uci.edu/~asuncion/pubs/KDD_08.pdf
    
    n_topics, vocabulary_size = topic_counts.shape

    for w, word in enumerate(words):
        topic = topic_assignment[w]
        document = document_assignment[w]

        topic_counts[topic, word] -= 1 #phi
        doc_counts[document,topic] -= 1 #theta
        topic_N[topic] -= 1
        
        P_topic_word = (topic_counts[:,word] + gamma)/(topic_N + vocabulary_size*gamma)
        P_document_topic = (doc_counts[document] + alpha)/(doc_N[document] + n_topics*alpha)

        P_Z = P_document_topic*P_topic_word
        P_Z /= np.sum(P_Z)

        new_topic = np.random.multinomial(1, P_Z).argmax()

        topic_assignment[w] = new_topic
        topic_counts[new_topic, word] += 1
        doc_counts[document, new_topic] += 1
        topic_N[new_topic] += 1
    
    return topic_assignment, topic_counts, doc_counts, topic_N
