import numpy as np
from pypolyagamma import PyPolyaGamma
from scipy.special import beta as beta_function, expit as sigmoid

pg = PyPolyaGamma()

# alpha=.1
n_topics = 10
n_words = 1000
sig_0 = 1.0
embedding_size = 100
p = .3
beta_0 = .01

f = np.random.rand(n_words, embedding_size)
f_outer = np.array([np.outer(f_v,f_v) for f_v in f])

b = np.random.binomial(1, p, (n_topics, n_words))
sig_I_lamb = sig_0**2 * np.eye(embedding_size)
lamb = np.random.multivariate_normal(np.zeros(embedding_size), sig_I_lamb, size=n_topics)
sig_I_c = sig_0**2 * np.eye(n_topics)
c = np.random.multivariate_normal(np.zeros(n_topics), sig_I_c).reshape((-1,1))

def get_pi(f, lamb, c):
    return lamb.dot(f.T) + c

pi = get_pi(f, lamb, c)

def sample_gamma(pi, gamma):
    # todo vectorize
    for k in range(pi.shape[0]):
        for v in range(pi.shape[1]):
            gamma[k,v] = pg.pgdraw(1, pi[k,v])
    return gamma

gamma = sample_gamma(pi, np.zeros(pi.shape))

sig_I_lamb_inv = sig_0**-2 * np.eye(embedding_size)

def sample_lambda_and_c(gamma, f, sig_I_lamb_inv, b, lamb, c):
    for k in range(n_topics):
        # sample lambda
        SIGMA_k = np.linalg.inv(f_outer.T.dot(gamma[k]) + sig_I_lamb_inv)
        b_cgam = (b[k] - .5 - c[k]*gamma[k])
        MU_k = SIGMA_k.dot(b_cgam.dot(f))
        lamb[k] = np.random.multivariate_normal(MU_k, SIGMA_k)
        # sample c
        sig_k = (np.sum(gamma[k]) + sig_0**-2)**-1
        mu_k = sig_k * np.sum(b_cgam)
        c[k] = np.random.normal(mu_k, sig_k)

def sample_b(word_index, b, n, pi):
    b_not_v_beta = np.sum(b, axis=1) - b[:, word_index] * beta_0
    num_a = b_not_v_beta + np.sum(n, axis=1)
    num_b = beta_0
    num = beta_function(num_a, num_b)
    denom = beta_function(b_not_v_beta, beta_0)
    activation = sigmoid(pi[:,word_index])
    p_1 = num * activation / denom
    p_0 = 1 - activation
    p = p_1 / (p_1 + p_0)
    return np.random.binomial(1, p)

# sample_lambda_and_c(gamma, f, sig_I_lamb_inv, b, lamb, c)
# print(lamb)
# print(c)

print(sample_b(7, b, np.random.randint(10, size=(n_topics, n_words)), pi))

#todo sample z and beta_0
