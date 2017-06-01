import gensim
import pandas as pd
import smart_open
import random
import re
import numpy
from gensim.models import ldamodel

# setting random seed to get the same results each time.
numpy.random.seed(1)

# read data
dataframe = pd.read_csv('movie_plots.csv')

texts = []

for line in dataframe.Plots:
	lowered = line.lower()
    words = re.findall(r'\w+', lowered, flags = re.UNICODE | re.LOCALE)
    texts.append(words)

dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# train model
model = ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=6)

all_topics = model.get_document_topics(corpus, minimum_probability=0, per_word_topics=True)

# create file for tensors
with open('lda_tensor.tsv','a') as w:
	for doc_topics, word_topics, phi_values in all_topics:
		for topics in doc_topics:
			w.write(str(topics[1])+ "\t")
			w.write("\n")

# create file for metadata
with open('lda_metadata.tsv','w') as w:
    w.write('Titles\tGenres\n')
    for i,j in zip(dataframe.Titles, dataframe.Genres):
        w.write("%s\t%s\n" % (i,j))

