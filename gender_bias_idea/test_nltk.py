import nltk
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# download
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

'''
# parse
text = "the little yellow dog barked at the cat"
sentence = nltk.word_tokenize(text)

# tag the words
sentence_tag = nltk.pos_tag(sentence)
print(sentence_tag)

# 定义分块语法
# 这个规则是说一个NP块由一个可选的限定词后面跟着任何数目的形容词然后是一个名词组成
# NP(名词短语块) DT(限定词) JJ(形容词) NN(名词)
grammar = "NP: {<DT>?<JJ>*<NN>}"

# 进行分块
cp = nltk.RegexpParser(grammar)
tree = cp.parse(sentence_tag)
tree.draw()
'''

###################

# https://nlpforhackers.io/textrank-text-summarization/

import numpy as np
 
def pagerank(A, eps=0.0001, d=0.85):
    P = np.ones(len(A)) / len(A)
    while True:
        new_P = np.ones(len(A)) * (1 - d) / len(A) + d * A.T.dot(P)
        delta = abs(new_P - P).sum()
        if delta <= eps:
            return new_P
        P = new_P



from nltk.corpus import brown, stopwords
from nltk.cluster.util import cosine_distance
 
def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)
 
# One out of 5 words differ => 0.8 similarity
print(sentence_similarity("This is a good sentence".split(), "This is a bad sentence".split()))
 
# One out of 2 non-stop words differ => 0.5 similarity
print(sentence_similarity("This is a good sentence".split(), "This is a bad sentence".split(), stopwords.words('english')))
 
# 0 out of 2 non-stop words differ => 1 similarity (identical sentences)
print(sentence_similarity("This is a good sentence".split(), "This is a good sentence".split(), stopwords.words('english')))
 
# Completely different sentences=> 0.0
print(sentence_similarity("This is a good sentence".split(), "I want to go to the market".split(), stopwords.words('english')))


import numpy as np
 
 
# Get a text from the Brown Corpus
sentences = brown.sents('ca01')
 
print(sentences)
# [[u'The', u'Fulton', u'County', u'Grand', u'Jury', u'said', u'Friday', u'an', u'investigation', u'of', u"Atlanta's", u'recent', u'primary', u'election', u'produced', u'``', u'no', u'evidence', u"''", u'that', u'any', u'irregularities', u'took', u'place', u'.'], [u'The', u'jury', u'further', u'said', u'in', u'term-end', u'presentments', u'that', u'the', u'City', u'Executive', u'Committee', u',', u'which', u'had', u'over-all', u'charge', u'of', u'the', u'election', u',', u'``', u'deserves', u'the', u'praise', u'and', u'thanks', u'of', u'the', u'City', u'of', u'Atlanta', u"''", u'for', u'the', u'manner', u'in', u'which', u'the', u'election', u'was', u'conducted', u'.'], ...]
 
print(len(sentences))  #  98
 
# get the english list of stopwords
stop_words = stopwords.words('english')
 
 
def build_similarity_matrix(sentences, stopwords=None):
    # Create an empty similarity matrix
    S = np.zeros((len(sentences), len(sentences)))
 
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue
 
            S[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)
 
    # normalize the matrix row-wise
    for idx in range(len(S)):
        S[idx] /= S[idx].sum()
 
    return S
 
S = build_similarity_matrix(sentences, stop_words)    
print(S)

from operator import itemgetter 
 
sentence_ranks = pagerank(S)
 
print(sentence_ranks)
 
# Get the sentences ordered by rank
ranked_sentence_indexes = [item[0] for item in sorted(enumerate(sentence_ranks), key=lambda item: -item[1])]
print(ranked_sentence_indexes)
 
# Suppose we want the 5 most import sentences
SUMMARY_SIZE = 5
SELECTED_SENTENCES = sorted(ranked_sentence_indexes[:SUMMARY_SIZE])
print(SELECTED_SENTENCES)
 
# Fetch the most important sentences
summary = itemgetter(*SELECTED_SENTENCES)(sentences)
 
# Print the actual summary
for sentence in summary:
    print(' '.join(sentence))



def textrank(sentences, top_n=5, stopwords=None):
    """
    sentences = a list of sentences [[w11, w12, ...], [w21, w22, ...], ...]
    top_n = how may sentences the summary should contain
    stopwords = a list of stopwords
    """
    S = build_similarity_matrix(sentences, stop_words) 
    sentence_ranks = pagerank(S)
 
    # Sort the sentence ranks
    ranked_sentence_indexes = [item[0] for item in sorted(enumerate(sentence_ranks), key=lambda item: -item[1])]
    selected_sentences = sorted(ranked_sentence_indexes[:top_n])
    summary = itemgetter(*selected_sentences)(sentences)
    return summary
 
for idx, sentence in enumerate(textrank(sentences, stopwords=stopwords.words('english'))):
    print("%s. %s" % ((idx + 1), ' '.join(sentence)))



    





