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

def preprocess_tweets(documents, vocabulary):
	tokenizer = RegexpTokenizer(r'\w+\'[a-z]+|\w+')

	texts = []
	for document in documents:
		clean_document = tweet_preprocessor.clean(document)
		clean_lower_document = clean_document.lower()
		tokens = tokenizer.tokenize(clean_lower_document)
		valid_tokens = [token for token in tokens if token in vocabulary]
		texts.append(tokens)
	return texts
