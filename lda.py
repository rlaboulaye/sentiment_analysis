import os
import time

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

def get_corpus(directory_path='./documents/txt_sentoken'):
	documents = get_documents(directory_path)
	preprocessed_documents = preprocess(documents)
	dictionary = corpora.Dictionary(preprocessed_documents)
	corpus = [dictionary.doc2bow(document) for document in preprocessed_documents]
	return dictionary, corpus

def initialize_parameters(topics, dictionary, corpus, alpha, beta):
	num_documents = len(corpus)
	num_words = len(dictionary)

	Z = []
	C_WT = np.zeros((num_words, topics))
	C_DT = np.zeros((num_documents, topics))
	for document_index, document in enumerate(corpus):
		Z_document = []
		for word_occurrence_tuple in document:
			word_index = word_occurrence_tuple[0]
			count = word_occurrence_tuple[1]
			for i in range(count):
				topic_assignment = np.random.randint(topics)
				Z_document.append([word_index, topic_assignment])
				C_WT[word_index, topic_assignment] += 1
				C_DT[document_index, topic_assignment] += 1
		Z.append(Z_document)

	#Z = np.random.randint(topics, size=(num_documents, num_words))
	# C_WT = np.zeros((num_words, topics))
	# C_DT = np.zeros((num_documents, topics))
	# for document_index in range(num_documents):
	# 	for word_index in range(num_words):
	# 		topic = Z[document_index, word_index]
	# 		# we chose to scale with co occurrence matrix
	# 		count = X[document_index, word_index]
	# 		C_WT[word_index, topic] += count
	# 		C_DT[document_index, topic] += count
	# C_WT = np.array([np.random.dirichlet(beta * np.ones(topics)) for i in range(num_words)])
	# C_DT = np.array([np.random.dirichlet(alpha * np.ones(topics)) for i in range(num_documents)])
	return Z, C_WT, C_DT 

def run_gensim_lda(topics, passes=100, num_words_to_display=20):
	dictionary, corpus = get_corpus()
	ldamodel = models.ldamodel.LdaModel(corpus, num_topics=topics, id2word = dictionary, passes=passes)
	print(ldamodel.print_topics(num_topics=topics, num_words=num_words_to_display))

def display_phi(phi, dictionary, num_words_to_display):
	for topic_index, topic in enumerate(phi.T):
		labelled_probabilities = [(dictionary[word_index], prob) for word_index, prob in enumerate(topic)]
		sorted_probabilities = sorted(labelled_probabilities, key=lambda x: x[1], reverse=True)[:num_words_to_display]
		print('Topic ' + str(topic_index) + ': ', sorted_probabilities)

def run_lda(topics, passes=100, alpha=.1, beta=.01, num_words_to_display=20):
	dictionary, corpus = get_corpus()
	num_documents = len(corpus)
	num_words = len(dictionary)
	Z, C_WT, C_DT = initialize_parameters(topics, dictionary, corpus, alpha, beta)
	for i in range(passes):
		print(gibbs_sampling(Z, C_WT, C_DT, alpha, beta))

	theta = (C_DT + alpha) / np.sum(C_DT+ alpha, axis=1)[:, None]
	phi = ((C_WT + beta) / np.sum((C_WT + beta), axis=0))
	display_phi(phi, dictionary, num_words_to_display)

if __name__ == "__main__":
	passes = 2
	start_time = time.time()
	run_gensim_lda(2, passes, num_words_to_display=50)
	print('Gensim time: ' + str(time.time() - start_time))
	start_time = time.time()
	run_lda(2, passes, num_words_to_display=50)
	print('Our time: ' + str(time.time() - start_time))