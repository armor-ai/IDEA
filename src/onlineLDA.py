# coding=utf-8
"""Online Latent Dirichlet allocation using collapsed Gibbs sampling"""

from __future__ import absolute_import, division, unicode_literals  # noqa
import logging
import sys

import numpy as np
import numbers

import _lda

logger = logging.getLogger('lda')

PY2 = sys.version_info[0] == 2
if PY2:
    range = xrange


class OLDA:
    """Latent Dirichlet allocation using collapsed Gibbs sampling

    Parameters
    ----------
    n_topics : int
        Number of topics

    n_iter : int, default 2000
        Number of sampling iterations

    alpha : float, default 0.1
        Dirichlet parameter for distribution over topics

    eta : float, default 0.01
        Dirichlet parameter for distribution over words

    random_state : int or RandomState, optional
        The generator used for the initial topics.

    Attributes
    ----------
    `components_` : array, shape = [n_topics, n_features]
        Point estimate of the topic-word distributions (Phi in literature)
    `topic_word_` :
        Alias for `components_`
    `nzw_` : array, shape = [n_topics, n_features]
        Matrix of counts recording topic-word assignments in final iteration.
    `ndz_` : array, shape = [n_samples, n_topics]
        Matrix of counts recording document-topic assignments in final iteration.
    `doc_topic_` : array, shape = [n_samples, n_features]
        Point estimate of the document-topic distributions (Theta in literature)
    `nz_` : array, shape = [n_topics]
        Array of topic assignment counts in final iteration.

    Examples
    --------
    >>> import numpy
    >>> X = numpy.array([[1,1], [2, 1], [3, 1], [4, 1], [5, 8], [6, 1]])
    >>> import lda
    >>> model = lda.LDA(n_topics=2, random_state=0, n_iter=100)
    >>> model.fit(X) #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    LDA(alpha=...
    >>> model.components_
    array([[ 0.85714286,  0.14285714],
           [ 0.45      ,  0.55      ]])
    >>> model.loglikelihood() #doctest: +ELLIPSIS
    -40.395...

    References
    ----------
    Blei, David M., Andrew Y. Ng, and Michael I. Jordan. "Latent Dirichlet
    Allocation." Journal of Machine Learning Research 3 (2003): 993–1022.

    Griffiths, Thomas L., and Mark Steyvers. "Finding Scientific Topics."
    Proceedings of the National Academy of Sciences 101 (2004): 5228–5235.
    doi:10.1073/pnas.0307752101.

    Wallach, Hanna, David Mimno, and Andrew McCallum. "Rethinking LDA: Why
    Priors Matter." In Advances in Neural Information Processing Systems 22,
    edited by Y.  Bengio, D. Schuurmans, J. Lafferty, C. K. I. Williams, and A.
    Culotta, 1973–1981, 2009.

    Buntine, Wray. "Estimating Likelihoods for Topic Models." In Advances in
    Machine Learning, First Asian Conference on Machine Learning (2009): 51–64.
    doi:10.1007/978-3-642-05224-8_6.

    """

    def __init__(self, n_topics, n_iter=2000, random_state=None,
                 refresh=10, window_size=1, theta=0.5):
        self.n_topics = n_topics
        self.n_iter = n_iter
        self.window_size = window_size
        # if random_state is None, check_random_state(None) does nothing
        # other than return the current numpy RandomState
        self.random_state = random_state
        self.refresh = refresh
        self.alpha_m = None
        self.eta_m = None
        self.eta_l = None
        self.alpha_sum = None
        self.eta_sum = None
        self.theta = theta
        self.alpha = 0.1
        self.B = []
        self.A = []
        self.loglikelihoods_pred = []
        self.loglikelihoods_train = []
        self.ll = -1
        # random numbers that are reused
        rng = self.check_random_state(random_state)
        self._rands = rng.rand(1024**2 // 8)  # 1MiB of random variates

        # configure console logging if not already configured
        if len(logger.handlers) == 1 and isinstance(logger.handlers[0], logging.NullHandler):
            logging.basicConfig(level=logging.INFO)

    def fit(self, X, alpha=0.1, eta=0.01, y=None):
        """Fit the model with X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features. Sparse matrix allowed.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # =================== online process===================

        # split X into time slots, feed into LDA model with alpha, beta matrix and B
        for t, x in enumerate(X):
            # if t == len(X) - 1:     # skip the last batch
            #     return self
            D, W = x.shape

            n_topics = self.n_topics
            if t == 0:
                eta_m = np.full((n_topics, W), eta).astype(np.float64)
            else:
                eta_m = self.soft_align(self.B, self.window_size, self.theta).astype(np.float64)
            alpha_m = np.full((D, n_topics), alpha).astype(np.float64)
            self.alpha_m = alpha_m
            self.eta_m = eta_m
            self.eta_l = eta_m
            self.alpha_sum = np.sum(alpha_m, 1)
            self.eta_sum = np.sum(eta_m, 1)
            self.alpha = alpha
            # fit the model
            self._fit(x, alpha_m, eta_m)
            # test the model
            # if t != len(X) - 1:
            #     ll_pred = self.estimate_ll(X[t+1])
            #     self.loglikelihoods_pred.append(ll_pred)
            self.loglikelihoods_train.append(self.ll)
            self.B.append(self.topic_word_)
            self.A.append(self.doc_topic_)
        return self

    def soft_align(self, B, window_size, theta):
        """
        Soft alignment to produce a soft weight sum of B according to window size
        """
        eta = B[-1]
        eta_new = np.zeros(eta.shape)
        weights = self.softmax(eta, B, window_size)
        for i in range(window_size):
            if i > len(B)-1:
                break
            B_i = B[-i-1] * weights[i][:, np.newaxis]
            eta_new += B_i
        eta_new = theta * self.eta_l + (1 - theta) * eta_new
        return eta_new

    def softmax(self, eta, B, window_size):
        prods = []
        for i in range(window_size):
            if i > len(B)-1:
                break
            prods.append(np.einsum('ij,ij->i', eta, B[-i-1]))
        weights = np.exp(np.array(prods))
        # weights = np.ones(weights.shape)            # compare to uniform
        n_weights = weights / np.sum(weights, 0)  # column normalize
        return n_weights

    def estimate_ll(self, X):
        doc_topic = self.transform(X)
        ll_pred = self.compute_loglikelihood(doc_topic, self.topic_word_, X)
        logging.info("test perplexity: %f"%ll_pred)
        return ll_pred

    # return the +ELLIPSIS
    def compute_loglikelihood(self, doc_topic, topic_word, X):
        temp = np.log(np.dot(doc_topic, topic_word))
        return np.sum(X.multiply(temp))

    def fit_transform(self, X, y=None):
        """Apply dimensionality reduction on X

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features. Sparse matrix allowed.

        Returns
        -------
        doc_topic : array-like, shape (n_samples, n_topics)
            Point estimate of the document-topic distributions

        """

        if isinstance(X, np.ndarray):
            # in case user passes a (non-sparse) array of shape (n_features,)
            # turn it into an array of shape (1, n_features)
            X = np.atleast_2d(X)
        self._fit(X)
        return self.doc_topic_

    def transform(self, X, max_iter=20, tol=1e-16):
        """Transform the data X according to previously fitted model

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.
        max_iter : int, optional
            Maximum number of iterations in iterated-pseudocount estimation.
        tol: double, optional
            Tolerance value used in stopping condition.

        Returns
        -------
        doc_topic : array-like, shape (n_samples, n_topics)
            Point estimate of the document-topic distributions

        Note
        ----
        This uses the "iterated pseudo-counts" approach described
        in Wallach et al. (2009) and discussed in Buntine (2009).

        """
        if isinstance(X, np.ndarray):
            # in case user passes a (non-sparse) array of shape (n_features,)
            # turn it into an array of shape (1, n_features)
            X = np.atleast_2d(X)
        doc_topic = np.empty((X.shape[0], self.n_topics))
        WS, DS = self.matrix_to_lists(X)
        # TODO: this loop is parallelizable
        for d in np.unique(DS):
            doc_topic[d] = self._transform_single(WS[DS == d], max_iter, tol)
        return doc_topic

    def _transform_single(self, doc, max_iter, tol):
        """Transform a single document according to the previously fit model

        Parameters
        ----------
        X : 1D numpy array of integers
            Each element represents a word in the document
        max_iter : int
            Maximum number of iterations in iterated-pseudocount estimation.
        tol: double
            Tolerance value used in stopping condition.

        Returns
        -------
        doc_topic : 1D numpy array of length n_topics
            Point estimate of the topic distributions for document

        Note
        ----

        See Note in `transform` documentation.

        """
        PZS = np.zeros((len(doc), self.n_topics))
        for iteration in range(max_iter + 1): # +1 is for initialization
            PZS_new = self.components_[:, doc].T
            PZS_new *= (PZS.sum(axis=0) - PZS + self.alpha)
            PZS_new /= PZS_new.sum(axis=1)[:, np.newaxis] # vector to single column matrix
            delta_naive = np.abs(PZS_new - PZS).sum()
            logger.debug('transform iter {}, delta {}'.format(iteration, delta_naive))
            PZS = PZS_new
            if delta_naive < tol:
                break
        theta_doc = PZS.sum(axis=0) / PZS.sum()
        assert len(theta_doc) == self.n_topics
        assert theta_doc.shape == (self.n_topics,)
        return theta_doc

    def _fit(self, X, alpha, eta):
        """Fit the model to the data X

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features. Sparse matrix allowed.
        """
        random_state = self.check_random_state(self.random_state)
        rands = self._rands.copy()
        self._initialize(X)


        for it in range(self.n_iter):
            # FIXME: using numpy.roll with a random shift might be faster
            random_state.shuffle(rands)
            if it % self.refresh == 0:
                ll = self.loglikelihood()
                logger.info("<{}> log likelihood: {:.0f}".format(it, ll))
                # keep track of loglikelihoods for monitoring convergence
                self.loglikelihoods_.append(ll)
            self._sample_topics(rands)
        self.ll = self.loglikelihood()
        logger.info("<{}> log likelihood: {:.0f}".format(self.n_iter - 1, self.ll))
        # note: numpy /= is integer division
        self.components_ = (self.nzw_ + eta).astype(float)
        self.components_ /= np.sum(self.components_, axis=1)[:, np.newaxis]
        self.topic_word_ = self.components_
        self.doc_topic_ = (self.ndz_ + alpha).astype(float)
        self.doc_topic_ /= np.sum(self.doc_topic_, axis=1)[:, np.newaxis]

        # delete attributes no longer needed after fitting to save memory and reduce clutter
        del self.WS
        del self.DS
        del self.ZS
        return self

    def _initialize(self, X):
        D, W = X.shape
        N = int(X.sum())
        n_topics = self.n_topics
        n_iter = self.n_iter
        logger.info("n_documents: {}".format(D))
        logger.info("vocab_size: {}".format(W))
        logger.info("n_words: {}".format(N))
        logger.info("n_topics: {}".format(n_topics))
        logger.info("n_iter: {}".format(n_iter))

        self.nzw_ = nzw_ = np.zeros((n_topics, W), dtype=np.intc)
        self.ndz_ = ndz_ = np.zeros((D, n_topics), dtype=np.intc)
        self.nz_ = nz_ = np.zeros(n_topics, dtype=np.intc)

        self.WS, self.DS = WS, DS = self.matrix_to_lists(X)
        self.ZS = ZS = np.empty_like(self.WS, dtype=np.intc)
        np.testing.assert_equal(N, len(WS))
        for i in range(N):
            w, d = WS[i], DS[i]
            z_new = i % n_topics
            ZS[i] = z_new
            ndz_[d, z_new] += 1
            nzw_[z_new, w] += 1
            nz_[z_new] += 1
        self.loglikelihoods_ = []

    def loglikelihood(self):
        """Calculate complete log likelihood, log p(w,z)

        Formula used is log p(w,z) = log p(w|z) + log p(z)
        """
        nzw, ndz, nz = self.nzw_, self.ndz_, self.nz_
        alpha_m = self.alpha_m
        eta_m = self.eta_m
        alpha_sum = self.alpha_sum
        eta_sum = self.eta_sum
        nd = np.sum(ndz, axis=1).astype(np.intc)
        return _lda._loglikelihood(nzw, ndz, nz, nd, alpha_m, eta_m, alpha_sum, eta_sum)

    def _sample_topics(self, rands):
        """Samples all topic assignments. Called once per iteration."""
        n_topics, vocab_size = self.nzw_.shape
        alpha = self.alpha_m    #np.repeat(self.alpha, n_topics).astype(np.float64)
        eta = self.eta_m        #np.repeat(self.eta, vocab_size).astype(np.float64)
        eta_sum = self.eta_sum
        _lda._sample_topics(self.WS, self.DS, self.ZS, self.nzw_, self.ndz_, self.nz_,
                                alpha, eta, eta_sum, rands)
        # self.sample_topics_py(self.WS, self.DS, self.ZS, self.nzw_, self.ndz_, self.nz_,
        #                         alpha, eta, eta_sum, rands)

    def searchsorted_py(self, arr, length, value):
        """Bisection search (c.f. numpy.searchsorted)

        Find the index into sorted array `arr` of length `length` such that, if
        `value` were inserted before the index, the order of `arr` would be
        preserved.
        """
        imin = 0
        imax = length
        while imin < imax:
            imid = imin + ((imax - imin) >> 2)
            if value > arr[imid]:
                imin = imid + 1
            else:
                imax = imid
        return imin

    def sample_topics_py(self, WS, DS, ZS, nzw, ndz, nz, alpha, eta, eta_sum, rands):

        N = WS.shape[0]

        n_rand = rands.shape[0]

        n_topics = nz.shape[0]
        # cdef double eta_sum = 0

        dist_sum = np.zeros(n_topics, dtype=float)

        # for i in range(eta.shape[0]):
        #    eta_sum += eta[i]

        for i in range(N):
            w = WS[i]
            d = DS[i]
            z = ZS[i]

            nzw[z, w] -= 1
            ndz[d, z] -= 1
            nz[z] -= 1

            dist_cum = 0
            for k in range(n_topics):
                # eta is a double so cdivision yields a double
                dist_cum += (nzw[k, w] + eta[k, w]) / (nz[k] + eta_sum[k]) * (ndz[d, k] + alpha[d, k])
                dist_sum[k] = dist_cum

            r = rands[i % n_rand] * dist_cum  # dist_cum == dist_sum[-1]
            z_new = self.searchsorted_py(dist_sum, n_topics, r)

            ZS[i] = z_new
            nzw[z_new, w] += 1
            ndz[d, z_new] += 1
            nz[z_new] += 1

    def check_random_state(self, seed):
        if seed is None:
            # i.e., use existing RandomState
            return np.random.mtrand._rand
        if isinstance(seed, (numbers.Integral, np.integer)):
            return np.random.RandomState(seed)
        if isinstance(seed, np.random.RandomState):
            return seed
        raise ValueError("{} cannot be used as a random seed.".format(seed))

    def matrix_to_lists(self, doc_word):
        """Convert a (sparse) matrix of counts into arrays of word and doc indices

        Parameters
        ----------
        doc_word : array or sparse matrix (D, V)
            document-term matrix of counts

        Returns
        -------
        (WS, DS) : tuple of two arrays
            WS[k] contains the kth word in the corpus
            DS[k] contains the document index for the kth word

        """
        if np.count_nonzero(doc_word.sum(axis=1)) != doc_word.shape[0]:
            logger.warning("all zero row in document-term matrix found")
        if np.count_nonzero(doc_word.sum(axis=0)) != doc_word.shape[1]:
            logger.warning("all zero column in document-term matrix found")
        sparse = True
        try:
            # if doc_word is a scipy sparse matrix
            doc_word = doc_word.copy().tolil()
        except AttributeError:
            sparse = False

        if sparse and not np.issubdtype(doc_word.dtype, int):
            raise ValueError("expected sparse matrix with integer values, found float values")

        ii, jj = np.nonzero(doc_word)
        if sparse:
            ss = tuple(doc_word[i, j] for i, j in zip(ii, jj))
        else:
            ss = doc_word[ii, jj]

        n_tokens = int(doc_word.sum())
        DS = np.repeat(ii, ss).astype(np.intc)
        WS = np.empty(n_tokens, dtype=np.intc)
        startidx = 0
        for i, cnt in enumerate(ss):
            cnt = int(cnt)
            WS[startidx:startidx + cnt] = jj[i]
            startidx += cnt
        return WS, DS

    def lists_to_matrix(self, WS, DS):
        """Convert array of word (or topic) and document indices to doc-term array

        Parameters
        -----------
        (WS, DS) : tuple of two arrays
            WS[k] contains the kth word in the corpus
            DS[k] contains the document index for the kth word

        Returns
        -------
        doc_word : array (D, V)
            document-term array of counts

        """
        D = max(DS) + 1
        V = max(WS) + 1
        doc_word = np.empty((D, V), dtype=np.intc)
        for d in range(D):
            for v in range(V):
                doc_word[d, v] = np.count_nonzero(WS[DS == d] == v)
        return doc_word