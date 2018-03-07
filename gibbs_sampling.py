import numpy as np

# X - document word co-occurrence matrix
# Z - word topic assignment
# C_WT - word topic matrix of counts
# C_DT - document topic matrix of counts
def gibbs_sampling(X, Z, C_WT, C_DT, alpha=.1, beta=.01):

	# Full conditional of Z
	for document_index in range(C_DT.shape[0]):
		for word_index in range(C_WT.shape[0]):
			count = X[document_index, word_index]
			if count > 0:

				# decrement C_WT and C_DT
				current_topic_assignment = Z[document_index, word_index]
				C_WT[word_index, current_topic_assignment] -= count
				C_DT[document_index, current_topic_assignment] -= count
				#

				theta_doc = (C_DT[document_index] + alpha) / np.sum(C_DT[document_index] + alpha)
				phi_word = (C_WT[word_index] + beta) / np.sum((C_WT + beta), axis=0)
				prob_dist_over_topics_for_word_in_doc = np.exp(np.log(phi_word) + np.log(theta_doc))
				Z[document_index, word_index] = np.random.multinomial(1, prob_dist_over_topics_for_word_in_doc).argmax()

				# update C_WT and C_DT
				current_topic_assignment = Z[document_index, word_index]
				C_WT[word_index, current_topic_assignment] += count
				C_DT[document_index, current_topic_assignment] += count
				#
