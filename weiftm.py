import os
import time
from io import StringIO
import itertools
import pickle

import numpy as np
from scipy.stats import bernoulli, dirichlet, norm
from scipy.special import beta as beta_function, expit as sigmoid
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from gensim import corpora, models
from pypolyagamma import PyPolyaGamma

from preprocess import preprocess_tweets


class WEIFTM():

    NO_TOPIC = -1

    def get_documents_from_directory(self, directory_path):
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

    def get_documents_from_csv(self, csv_path, text_name="text", class_name="class"):
        with open(csv_path, 'r', encoding='utf8', errors='ignore') as csv_file:
            dataframe = pd.read_csv(StringIO(csv_file.read()))[:20]
            dataframe = dataframe.fillna(value={class_name: ''})
            dataframe[class_name] = LabelEncoder().fit_transform(dataframe[class_name])
            self.labels = dict(dataframe[class_name])
            return list(dataframe[text_name])

    def get_embedding_vocabulary(self, embedding_path):
        vocabulary = set()
        with open(embedding_path) as emb_file:
            for line in emb_file:
                if line != "":
                    word = line.strip().split(" ", 1)[0]
                    vocabulary.add(word)
        return vocabulary

    def load_corpus(self, documents, vocabulary, custom_stop_words=[]):
        preprocessed_documents = preprocess_tweets(documents, vocabulary, custom_stop_words)
        self.dictionary = corpora.Dictionary(preprocessed_documents)
        self.n_words = len(self.dictionary)
        self.corpus = [self.dictionary.doc2bow(document) for document in preprocessed_documents]
        self.n_documents = len(self.corpus)

    def load_embeddings(self, embedding_size, embedding_path, corpus_dir):
        self.embedding_size = embedding_size
        cache_dir = "./cache/{}/".format(corpus_dir.strip(os.path.sep).strip('.csv').split(os.path.sep)[-1])
        embedding_cache_path = cache_dir + "embedding{}.npy".format(embedding_size)
        if os.path.isfile(embedding_cache_path):
            self.f = np.load(embedding_cache_path)
        else:
            vocabulary = set(self.dictionary.values())
            self.f = np.empty((self.n_words, self.embedding_size))
            with open(embedding_path) as emb_file:
                for line in emb_file:
                    if line != "":
                        word, str_embedding = line.strip().split(" ", 1)
                        if word in vocabulary:
                            word_index = self.dictionary.token2id[word]
                            self.f[word_index] = np.array(str_embedding.split(" "), dtype=float)
            if not os.path.isdir(cache_dir):
                os.makedirs(cache_dir)
            np.save(embedding_cache_path, self.f)
        self.f_outer = np.array([np.outer(f_v,f_v) for f_v in self.f])

    def _initialize_parameters(self, n_topics, topic_sparsity):
        self._init_b(n_topics, topic_sparsity)
        self._init_n_m_Z(n_topics)
        self._init_lamb(n_topics)
        self._init_c(n_topics)
        self._init_pi()
        self._init_embedding_aux_params(n_topics)

    def _init_b(self, n_topics, topic_sparsity):
        self.b = np.random.binomial(1, topic_sparsity, (n_topics, self.n_words))
        self.b_sum_ax1 = np.sum(self.b, axis=1)

    def _init_n_m_Z(self, n_topics):
        assert(hasattr(self, "b"))

        self.n = np.zeros((n_topics, self.n_words))
        self.m = np.zeros((self.n_documents, n_topics))
        self.Z = []
        for document_index, document in enumerate(self.corpus):
            Z_document = []
            for word_occurrence_tuple in document:
                word_index = word_occurrence_tuple[0]
                count = word_occurrence_tuple[1]
                for _ in range(count):
                    nonzero_b = self.b[:, word_index].nonzero()[0]
                    if len(nonzero_b) == 0:
                        topic_assignment = WEIFTM.NO_TOPIC
                    else:
                        topic_assignment = np.random.choice(nonzero_b)
                        self.n[topic_assignment, word_index] += 1
                        self.m[document_index, topic_assignment] += 1
                    Z_document.append([word_index, topic_assignment])
            self.Z.append(Z_document)

    def _init_lamb(self, n_topics):
        sig_I_lamb = self.sig_0**2 * np.eye(self.embedding_size)
        self.lamb = np.random.multivariate_normal(np.zeros(self.embedding_size), sig_I_lamb, size=n_topics)
        self.sig_I_lamb_inv = self.sig_0**-2 * np.eye(self.embedding_size)

    def _init_c(self, n_topics):
        sig_I_c = self.sig_0**2 * np.eye(n_topics)
        self.c = np.random.multivariate_normal(np.zeros(n_topics), sig_I_c).reshape((-1,1))

    def _init_pi(self):
        self.pi = np.matmul(self.lamb, self.f.T) + self.c

    def _init_embedding_aux_params(self, n_topics):
        self.gamma = np.empty((n_topics, self.n_words))
        self.gamma_sum_ax1 = np.zeros(n_topics)
        self.SIGMA_inv = np.empty((n_topics, self.embedding_size, self.embedding_size))
        self.b_cgam = np.empty((n_topics, self.n_words))
        self.b_cgam_sum_ax1 = np.zeros(n_topics)
        self.MU = np.empty((n_topics, self.embedding_size))
        for k in range(n_topics):
            for word_index in range(self.n_words):
                self.gamma[k,word_index] = self.pg.pgdraw(1, self.pi[k,word_index])
                self.gamma_sum_ax1[k] += self.gamma[k,word_index]

            self.SIGMA_inv[k] = np.matmul(self.f_outer.T, self.gamma[k]) + self.sig_I_lamb_inv
            self.b_cgam[k] = self.b[k] - .5 - self.c[k]*self.gamma[k]
            self.b_cgam_sum_ax1[k] = np.sum(self.b_cgam[k])

        self.b_cgam_f = np.matmul(self.b_cgam, self.f)
        for k in range(n_topics):
            SIGMA_k = np.linalg.inv(self.SIGMA_inv[k])
            self.MU[k] = np.matmul(SIGMA_k, self.b_cgam_f[k])

    def train(self, n_topics, topic_sparsity=.3, alpha_0=.1, beta_0=.01, delta_0=1, sig_0=1, iters=10):
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.delta_0 = delta_0
        self.sig_0 = sig_0
        self.pg = PyPolyaGamma()
        self._initialize_parameters(n_topics, topic_sparsity)
        self.log_likelihoods = []
        for i in range(iters):
            # start_time = time.time()
            self._gibbs_sample(n_topics)
            # print("gibbs", time.time() - start_time)

            # start_time = time.time()
            self.log_likelihoods.append(self._compute_total_log_likelihood(n_topics))
            # print("log_likelihood", time.time() - start_time)
        return self.log_likelihoods

    def _gibbs_sample(self, n_topics):
        # gibbs_iter_time = time.time()
        for document_index, Z_document in enumerate(self.Z):
            document_length = len(Z_document)
            for token_index, Z_token_pair in enumerate(Z_document):

                # print("gibbs iter", time.time() - gibbs_iter_time)
                # gibbs_iter_time = time.time()
                # print(token_index, "/", document_length, document_index, "/", self.n_documents)

                word_index = Z_token_pair[0]
                topic_assignment = Z_token_pair[1]
                if topic_assignment != WEIFTM.NO_TOPIC:
                    self.n[topic_assignment, word_index] -= 1
                    self.m[document_index, topic_assignment] -= 1

                # start_time = time.time()
                self._sample_b(word_index)
                # print("sample_b", time.time() - start_time)

                # start_time = time.time()
                topic_assignment = self._sample_z(document_index, word_index)
                # print("sample_z", time.time() - start_time)
                Z_token_pair[1] = topic_assignment

                if topic_assignment != WEIFTM.NO_TOPIC:
                    self.n[topic_assignment, word_index] += 1
                    self.m[document_index, topic_assignment] += 1

                # start_time = time.time()
                self._sample_embeddings(n_topics, word_index)
                # print("sample_embeddings", time.time() - start_time)

    def _sample_b(self, word_index):
        b_not_v = self.b_sum_ax1 - self.b[:, word_index]

        b_not_v[b_not_v == 0] += self.delta_0
        b_not_v_beta = b_not_v * self.beta_0

        num_a = b_not_v_beta + np.sum(self.n, axis=1)
        num_b = self.beta_0
        num = beta_function(num_a, num_b)
        denom = beta_function(b_not_v_beta, self.beta_0)
        activation = sigmoid(self.pi[:,word_index])
        p_1 = num * activation / denom
        p_0 = 1 - activation
        p = p_1 / (p_1 + p_0)

        self.b_sum_ax1 -= self.b[:, word_index]
        self.b[:, word_index] |= np.random.binomial(1, p)
        self.b_sum_ax1 += self.b[:, word_index]

    def _sample_z(self, document_index, word_index):
        if self.b[:,word_index].sum() == 0:
            topic_assignment = WEIFTM.NO_TOPIC
        else:
            p = (self.alpha_0 + self.m[document_index]) * (self.n[:,word_index].flatten() + self.beta_0) / (self.n[:,word_index] + self.beta_0).sum() * self.b[:,word_index]
            p /= p.sum()
            topic_assignment = np.random.multinomial(1, p).argmax()
        return topic_assignment

    def _sample_embeddings(self, n_topics, word_index):
        for k in range(n_topics):
            # sample gamma
            old_gamma_k_word_index = self.gamma[k,word_index]
            self.gamma[k,word_index] = self.pg.pgdraw(1, self.pi[k,word_index])
            self.gamma_sum_ax1[k] += self.gamma[k,word_index] - old_gamma_k_word_index

            # sample lamb
            self.SIGMA_inv[k] += (self.gamma[k,word_index] - old_gamma_k_word_index) * self.f_outer[word_index]
            SIGMA_k = np.linalg.inv(self.SIGMA_inv[k])

            old_b_cgam_k_word_index = self.b_cgam[k, word_index]
            self.b_cgam[k, word_index] = self.b[k, word_index] - .5 - self.c[k]*self.gamma[k, word_index]
            self.b_cgam_sum_ax1[k] += self.b_cgam[k, word_index] - old_b_cgam_k_word_index

            self.b_cgam_f[k] = self.b_cgam[k, word_index] * self.f[word_index]
            self.MU[k] = np.matmul(SIGMA_k, self.b_cgam_f[k])

            self.lamb[k] = np.random.multivariate_normal(self.MU[k], SIGMA_k)

            # sample c
            sig_k = (self.gamma_sum_ax1[k] + self.sig_0**-2)**-1
            mu_k = sig_k * self.b_cgam_sum_ax1[k]
            self.c[k] = np.random.normal(mu_k, sig_k)

        # update pi
        self.pi = np.matmul(self.lamb, self.f.T) + self.c

    def _compute_total_log_likelihood(self, n_topics):
        log_likelihood = 0

        theta = self.get_theta()
        log_theta = np.log(theta)
        phi = self.get_phi()
        log_phi = np.log(phi)

        ALPHA = self.alpha_0 * np.ones(n_topics)

        for document_index in range(self.n_documents):
            # theta
            log_likelihood += np.log(dirichlet.pdf(theta[document_index], ALPHA))

            for token_index in range(len(self.Z[document_index])):
                word_index, topic_index = self.Z[document_index][token_index]
                if topic_index != self.NO_TOPIC:
                    # w
                    log_likelihood += log_phi[topic_index, word_index]
                    # z
                    log_likelihood += log_theta[document_index, topic_index]

        log_likelihood += np.sum(np.log(bernoulli.pmf(self.b, sigmoid(self.pi))))

        for k in range(n_topics):
            # phi
            b_k_nonzero = self.b[k].nonzero()[0]
            BETA = self.beta_0 * np.ones(b_k_nonzero.shape[0])
            log_likelihood += np.log(dirichlet.pdf(phi[k][b_k_nonzero], BETA))
            # c
            log_likelihood += np.log(norm.pdf(self.c[k], 0, self.sig_0))

            for l in range(self.embedding_size):
                # lamb
                log_likelihood += np.log(norm.pdf(self.lamb[k, l], 0, self.sig_0))

        return log_likelihood

    def get_phi(self):
        n_b = (self.n + self.beta_0) * self.b
        return n_b / n_b.sum(axis=1).reshape(-1, 1)

    def get_theta(self):
        return (self.m + self.alpha_0) / (self.m + self.alpha_0).sum(axis=1).reshape(-1, 1)

    def print_phi(self, n_words):
        phi = self.get_phi()
        for topic_index, topic, in enumerate(phi):
            labelled_probabilities = [(self.dictionary[word_index], prob) for word_index, prob in enumerate(topic)]
            sorted_probabilities = sorted(labelled_probabilities, key=lambda x: x[1], reverse=True)[:n_words]
            print('Topic {}:'.format(topic_index), sorted_probabilities)

    def print_theta(self):
        theta = self.get_theta()
        for document_index, document in enumerate(theta):
            print('Document {}:'.format(document_index), document)

    def get_classification_accuracy(self, n_topics):
        theta = self.get_theta()
        predictions = [distribution.argmax() for distribution in theta]
        prediction_set = set(predictions)
        label_set = set(self.labels.values())
        accuracies = []
        for tup in itertools.permutations(label_set, len(prediction_set)):
            count = 0.
            for index in self.labels:
                if self.labels[index] == tup[predictions[index]]:
                    count += 1.
            accuracies.append(count / len(predictions))
        print(predictions)
        print(accuracies)
        return max(accuracies)

    def plot_log_likelihoods(self, log_likelihoods, path):
        title = path.strip(os.path.sep).strip('.csv').split(os.path.sep)[-1]
        plt.title(title)
        plt.xlabel('epoch')
        plt.ylabel('log likehood')
        plt.plot(log_likelihoods)
        plt.show()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("pg")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def save(self, path):
        pickle.dump(self, open(path, "wb"))

    @staticmethod
    def load(path):
        return pickle.load(open(path, "rb"))


def main():
    n_topics = 3
    embedding_size = 50
    train_iters = 3
    custom_stop_words = ['_', 'link']
    path = "./documents/csv/global_warming_tweets.csv"
    # path = "./documents/txt_sentoken/"
    # path = "./documents/toy/"
    embedding_path = "./glove.6B/glove.6B.{}d.txt".format(embedding_size)

    start_time = time.time()
    weiftm = WEIFTM()
    embedding_vocabulary = weiftm.get_embedding_vocabulary(embedding_path)

    documents = weiftm.get_documents_from_csv(path)
    # documents = weiftm.get_documents_from_directory(path)

    weiftm.load_corpus(documents, embedding_vocabulary, custom_stop_words)
    weiftm.load_embeddings(embedding_size, embedding_path, path)

    load_time = time.time() - start_time

    start_time = time.time()
    log_likelihoods = weiftm.train(n_topics, iters=train_iters)
    train_time = time.time() - start_time

    pickle_path = path.strip("/").rsplit("/", 1)[-1] + ".p"
    weiftm.save(pickle_path)
    weiftm2 = WEIFTM.load(pickle_path)

    print("load time: {}".format(load_time))
    print("train time: {}".format(train_time))

    # weiftm.print_theta()
    print(list(weiftm.labels.values()))
    print(weiftm.get_classification_accuracy(n_topics))
    # weiftm.print_phi(100)
    # np.set_printoptions(threshold=np.nan)
    # print(weiftm.b)


    # weiftm.plot_log_likelihoods(log_likelihoods, path)


if __name__ == '__main__':
    main()
