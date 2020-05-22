from glove.embedding import GloveEmbedding
from glove.dataset import PairedDataset, PairedPolitical
from glove.visualize import GloveVisualize

PARSE_MODE = "phrase"
POLARITY_DIM = 1

glove_embedding = GloveEmbedding()


political_pair_data = ("../data/pairs/democratic_2005.txt", "../data/pairs/republican_2005.txt")
hashtag_pair_data = "../data/hashtag_pairs.csv"
# the pair info
polarity_words = PairedPolitical(political_pair_data, hashtag_pair_data, parse_mode=PARSE_MODE)
polarity_words.filter(glove_embedding)
# the visualizer
visualizer = GloveVisualize(glove_embedding, glove_embedding, polarity_words=polarity_words, polarity_dimensions=POLARITY_DIM)

# print(glove_embedding[["hello", "world"]])

'''
labeled_datafile = ["../data/democratic_cleaned.txt", "../data/republican_cleaned.txt"]
datasets = PairedDataset([open(fname, encoding="utf8").read() for fname in labeled_datafile], parse_phrase=polarity_words.phrases)
'''

if __name__ == '__main__':
    VISUALIZE = True
    DEBUG = False
    if VISUALIZE:
        # visualizer.visualize_loss(loss_values, debug=DEBUG)
        visualizer.visualize_embedding(mode="polarity", debug=DEBUG)
        visualizer.visualize_embedding(mode="semantic", debug=DEBUG)
        visualizer.visualize_embedding(debug=DEBUG)

