from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
import preprocessor as tweet_preprocessor

def preprocess(documents):

	tokenizer = RegexpTokenizer(r'\w+\'[a-z]+|\w+')
	en_stop = get_stop_words('en')
	p_stemmer = PorterStemmer()

	texts = []
	for document in documents:
		raw = document.lower()
		tokens = tokenizer.tokenize(raw)
		stopped_tokens = [i for i in tokens if not i in en_stop]
		stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
		texts.append(stemmed_tokens)

	return texts

def preprocess_tweets(documents, vocabulary, custom_stop_words):
	tokenizer = RegexpTokenizer(r'\w+\'[a-z]+|\w+')
	en_stop = get_stop_words('en')
	en_stop.extend(custom_stop_words)
	p_stemmer = PorterStemmer()

	stem_to_possible_words = {}
	stem_to_word = {}
	texts = []
	for document in documents:
		clean_document = tweet_preprocessor.clean(document)
		clean_lower_document = clean_document.lower()
		tokens = tokenizer.tokenize(clean_lower_document)
		stopped_tokens = [token for token in tokens if not token in en_stop]
		valid_tokens = [token for token in stopped_tokens if token in vocabulary]
		stemmed_tokens = []
		for token in valid_tokens:
			stem = p_stemmer.stem(token)
			if stem not in stem_to_possible_words:
				stem_to_possible_words[stem] = []
			stem_to_possible_words[stem].append(token)
			stemmed_tokens.append(stem)
		if len(stemmed_tokens) > 0:
			texts.append(stemmed_tokens)
	for stem in stem_to_possible_words:
		possible_words = stem_to_possible_words[stem]
		stem_to_word[stem] = max(set(possible_words), key=possible_words.count)
	texts = [[stem_to_word[token] for token in text] for text in texts]
	return texts
