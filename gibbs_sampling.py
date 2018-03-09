import numpy as np

# X - document word co-occurrence matrix
# Z - word topic assignment
# C_WT - word topic matrix of counts
# C_DT - document topic matrix of counts
def gibbs_sampling(Z, C_WT, C_DT, alpha=.1, beta=.01):
	probabilities = []
	num_topics = C_WT.shape[1]
	prob_word_under_topic_denominator = np.sum(C_WT + beta, axis=0)
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

			prob_word_under_topic_denominator[current_topic_assignment] -= 1

			prob_dist_over_topics = (prob_word_under_topic_numerator / prob_word_under_topic_denominator) * \
				(prob_topic_under_document_numerator / prob_topic_under_document_denominator)
			Z_token_pair[1] = np.random.multinomial(1, prob_dist_over_topics / np.sum(prob_dist_over_topics)).argmax()

			current_topic_assignment = Z_token_pair[1]
			probabilities.append(prob_dist_over_topics[current_topic_assignment])
			C_WT[word_index, current_topic_assignment] += 1
			C_DT[document_index, current_topic_assignment] += 1
			prob_word_under_topic_denominator[current_topic_assignment] += 1

	return np.sum(np.log(probabilities))