import os

import numpy as np
from gensim import corpora, models

from preprocess import preprocess
from gibbs_sampling import gibbs_sampling

def get_documents(directory_path):
	documents = []
	for (path,dirs,files) in os.walk(directory_path):
		files.sort()
		for file_path in files:
			if file_path.endswith('.txt'):
				document_path = os.path.join(path, file_path)
				try:
					file = open(document_path, 'r')
					document = file.read()
					file.close()
					documents.append(document)
				except Exception as e:
					print(e)
	return documents

def get_corpus(directory_path='./documents'):
	documents = get_documents(directory_path)
	preprocessed_documents = preprocess(documents)
	dictionary = corpora.Dictionary(preprocessed_documents)
	corpus = [dictionary.doc2bow(document) for document in preprocessed_documents]
	return dictionary, corpus

def initialize_parameters(topics, X, alpha, beta):
	num_documents = X.shape[0]
	num_words = X.shape[1]
	Z = np.random.randint(topics, size=(num_documents, num_words))
	C_WT = np.zeros((num_words, topics))
	C_DT = np.zeros((num_documents, topics))
	for document_index in range(num_documents):
		for word_index in range(num_words):
			topic = Z[document_index, word_index]
			# we chose to scale with co occurrence matrix
			count = X[document_index, word_index]
			C_WT[word_index, topic] += count
			C_DT[document_index, topic] += count
	# C_WT = np.array([np.random.dirichlet(beta * np.ones(topics)) for i in range(num_words)])
	# C_DT = np.array([np.random.dirichlet(alpha * np.ones(topics)) for i in range(num_documents)])
	return Z, C_WT, C_DT 

def run_gensim_lda(topics, passes=100):
	dictionary, corpus = get_corpus()
	ldamodel = models.ldamodel.LdaModel(corpus, num_topics=topics, id2word = dictionary, passes=passes)
	print(ldamodel.print_topics(num_topics=topics, num_words=4))

def run_lda(topics, passes=100, alpha=.1, beta=.01):
	dictionary, corpus = get_corpus()
	print(dictionary)
	num_documents = len(corpus)
	num_words = len(dictionary)
	X = np.zeros(shape=(num_documents, num_words))
	for document_index in range(num_documents):
		for word_occurrence_tuple in corpus[document_index]:
			X[document_index, word_occurrence_tuple[0]] = word_occurrence_tuple[1]
	Z, C_WT, C_DT = initialize_parameters(topics, X, alpha, beta)
	for i in range(passes):
		gibbs_sampling(X, Z, C_WT, C_DT, alpha, beta)

	theta = (C_DT + alpha) / np.sum(C_DT+ alpha, axis=1)[:, None]
	phi = ((C_WT + beta) / np.sum((C_WT + beta), axis=0)).T
	print(phi)

if __name__ == "__main__":
	run_gensim_lda(2, 1)
	#run_lda(2)