import os
import time

import numpy as np
from gensim import corpora, models

from preprocess import preprocess


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

def get_corpus(directory_path='./documents/txt_sentoken/'):
	documents = get_documents(directory_path)
	preprocessed_documents = preprocess(documents)
	dictionary = corpora.Dictionary(preprocessed_documents)
	corpus = [dictionary.doc2bow(document) for document in preprocessed_documents]
	return dictionary, corpus

def run_gensim_lda(topics, passes=100, num_words_to_display=20):
	dictionary, corpus = get_corpus()
	ldamodel = models.ldamodel.LdaModel(corpus, num_topics=topics, id2word = dictionary, passes=passes)
	print(ldamodel.print_topics(num_topics=topics, num_words=num_words_to_display))

class LDA():

	def __init__(self):
		pass

	def load_corpus(self, directory="./documents/txt_sentoken/"):
		documents = get_documents(directory)
		preprocessed_documents = preprocess(documents)
		self.dictionary = corpora.Dictionary(preprocessed_documents)
		self.n_words = len(self.dictionary)
		self.corpus = [self.dictionary.doc2bow(document) for document in preprocessed_documents]
		self.n_documents = len(self.corpus)

	def _init_parameters(self, n_topics):
		self.Z = []
		self.C_WT = np.zeros((self.n_words, n_topics))
		self.C_DT = np.zeros((self.n_documents, n_topics))
		for document_index, document in enumerate(self.corpus):
			Z_document = []
			for word_occurrence_tuple in document:
				word_index = word_occurrence_tuple[0]
				count = word_occurrence_tuple[1]
				for _ in range(count):
					topic_assignment = np.random.randint(n_topics)
					Z_document.append([word_index, topic_assignment])
					self.C_WT[word_index, topic_assignment] += 1
					self.C_DT[document_index, topic_assignment] += 1
			self.Z.append(Z_document)

	def train(self, n_topics, alpha=.1, beta=.01, iters=100):
		self.alpha = alpha
		self.beta = beta
		self._init_parameters(n_topics)
		log_probs = []
		log_probs.append(self._compute_log_prob())
		self.prob_word_under_topic_denominator = np.sum(self.C_WT + beta, axis=0)
		for _ in range(iters):
			self._gibbs_sample(n_topics)
			log_probs.append(self._compute_log_prob())
		return log_probs

	def _gibbs_sample(self, n_topics):
		for document_index, Z_document in enumerate(self.Z):
			document_length = len(Z_document)
			prob_topic_under_document_denominator = document_length + n_topics * self.alpha
			for Z_token_pair in Z_document:
				word_index = Z_token_pair[0]
				current_topic_assignment = Z_token_pair[1]

				self.C_WT[word_index, current_topic_assignment] -= 1
				self.C_DT[document_index, current_topic_assignment] -= 1
				self.prob_word_under_topic_denominator[current_topic_assignment] -= 1

				prob_topic_under_document_numerator = self.C_DT[document_index] + self.alpha
				prob_word_under_topic_numerator = self.C_WT[word_index] + self.beta
				prob_dist_over_topics = (prob_word_under_topic_numerator / self.prob_word_under_topic_denominator) * \
					(prob_topic_under_document_numerator / prob_topic_under_document_denominator)

				current_topic_assignment = np.random.multinomial(1, prob_dist_over_topics / np.sum(prob_dist_over_topics)).argmax()
				Z_token_pair[1] = current_topic_assignment

				self.C_WT[word_index, current_topic_assignment] += 1
				self.C_DT[document_index, current_topic_assignment] += 1
				self.prob_word_under_topic_denominator[current_topic_assignment] += 1

	def get_theta(self):
		return (self.C_DT + self.alpha) / (self.C_DT+ self.alpha).sum(axis=1).reshape((-1, 1))

	def get_phi(self):
		return (self.C_WT + self.beta) / (self.C_WT + self.beta).sum(axis=0)

	def print_phi(self, n_words):
		phi = self.get_phi()
		for topic_index, topic, in enumerate(phi.T):
			labelled_probabilities = [(self.dictionary[word_index], prob) for word_index, prob in enumerate(topic)]
			sorted_probabilities = sorted(labelled_probabilities, key=lambda x: x[1], reverse=True)[:n_words]
			print('Topic {}:'.format(topic_index), sorted_probabilities)

	def _compute_log_prob(self):
		log_theta = np.log(self.get_theta())
		log_phi = np.log(self.get_phi())
		log_prob = 0
		for document_index in range(len(self.Z)):
			for j in range(len(self.Z[document_index])):
				word_index, topic_index = self.Z[document_index][j]
				log_prob += log_theta[document_index, topic_index] + log_phi[word_index, topic_index]
		return log_prob

if __name__ == "__main__":
	n_passes = 3
	n_topics = 2
	n_words_to_display = 50

	start_time = time.time()
	run_gensim_lda(n_topics, n_passes, num_words_to_display=50)
	print('Gensim time: ' + str(time.time() - start_time))

	# start_time = time.time()
	# run_lda(n_topics, n_passes, num_words_to_display=n_words_to_display)
	# print('Our time: ' + str(time.time() - start_time))

	start_time = time.time()
	lda = LDA()
	lda.load_corpus()
	print(lda.train(n_topics, iters=n_passes))
	lda.print_phi(n_words_to_display)
	print('Our time: ' + str(time.time() - start_time))
