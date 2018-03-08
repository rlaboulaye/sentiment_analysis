import numpy as np

# X - document word co-occurrence matrix
# Z - word topic assignment
# C_WT - word topic matrix of counts
# C_DT - document topic matrix of counts
def gibbs_sampling(Z, C_WT, C_DT, alpha=.1, beta=.01):
	num_topics = C_WT.shape[1]
	for document_index, Z_document in enumerate(Z):
		document_length = len(Z_document)
		prob_topic_under_document_denominator = document_length + num_topics * alpha
		for Z_token_pair in Z_document:
			word_index = Z_token_pair[0]
			current_topic_assignment = Z_token_pair[1]
			C_WT[word_index, current_topic_assignment] -= 1
			C_DT[document_index, current_topic_assignment] -= 1

			prob_topic_under_document_numerator = C_DT[document_index] + alpha
			prob_word_under_topic_numerator = C_WT[word_index] + beta
			prob_word_under_topic_denominator = np.sum(C_WT + beta, axis=0)
			prob_dist_over_topics = (prob_word_under_topic_numerator / prob_word_under_topic_denominator) * \
				(prob_topic_under_document_numerator / prob_topic_under_document_denominator)
			Z_token_pair[1] = np.random.multinomial(1, prob_dist_over_topics).argmax()

			current_topic_assignment = Z_token_pair[1]
			C_WT[word_index, current_topic_assignment] += 1
			C_DT[document_index, current_topic_assignment] += 1

	# for document_index in range(C_DT.shape[0]):
	# 	for word_index in range(C_WT.shape[0]):
	# 		count = X[document_index, word_index]
	# 		if count > 0:

	# 			# decrement C_WT and C_DT
	# 			current_topic_assignment = Z[document_index, word_index]
	# 			C_WT[word_index, current_topic_assignment] -= count
	# 			C_DT[document_index, current_topic_assignment] -= count
	# 			#

	# 			theta_doc = (C_DT[document_index] + alpha) / np.sum(C_DT[document_index] + alpha)
	# 			phi_word = (C_WT[word_index] + beta) / np.sum((C_WT + beta), axis=0)
	# 			prob_dist_over_topics_for_word_in_doc = np.exp(np.log(phi_word) + np.log(theta_doc))
	# 			Z[document_index, word_index] = np.random.multinomial(1, prob_dist_over_topics_for_word_in_doc).argmax()

	# 			# update C_WT and C_DT
	# 			current_topic_assignment = Z[document_index, word_index]
	# 			C_WT[word_index, current_topic_assignment] += count
	# 			C_DT[document_index, current_topic_assignment] += count
	# 			#
