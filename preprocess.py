from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer

def preprocess(documents):

	tokenizer = RegexpTokenizer(r'\w+')
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