import numpy as np


# verified
def circ_operation(X):
    N, m = X.shape
    # n = (N + 1) / 2
    Z = np.zeros(((m ** 2) * N, N))
    for i in range(m):
        for j in range(m):
            for k in range(N):
                cur_row = k + N * j + m * N * i
                Z[cur_row, :] = X[:, i] * np.roll(X[:, j], -k)
    return Z


def project_operation(Z1, desired_bound):
    rows, n = Z1.shape
    m = np.sqrt(rows / n)
    # m ,n here may not be the true m,n dimentions
    Zd = np.zeros_like(Z1)
    mask = Z1 != 0
    for i in range(Z1.shape[0]):
        row_sum = Z1[i, :].sum()  # alpha1
        support_len = mask[i, :].sum()  # mu.T*1
        if (i % ((m + 1) * n) == 0):
            Zd[i, mask[i, :]] = (row_sum - 1) / support_len
        else:
            if np.abs(row_sum) > desired_bound:
                Zd[i, mask[i, :]] = (row_sum - desired_bound) / support_len

    Z2 = Z1 - Zd
    # assert ((Zd==0)==(mask==0)).all()
    assert ((Z2 == 0) == (Z1 == 0)).all()
    return Z2


def vectorize_operation1(X):
    return X.T.reshape(-1)[np.newaxis, :].T


# verified
def vectorize_operation3(Z):
    N = Z.shape[1]
    m = int(np.sqrt(Z.shape[0] / N))
    vec_mat = np.zeros((N * m, N * m))
    for k in range(m):
        for i in range(m):
            for j in range(N):
                vec_mat[j + k * N, i * N:(i + 1) * N] = np.roll(Z[i * N + k * m * N:(i + 1) * N + k * m * N, j], j)
    return vec_mat


def create_dict(iterations=10000, n=64, m=2, desired_bound=0.05, verbose=False):
    N = 2*n-1

    x = np.random.randn(n, m)
    x /= np.sqrt(np.sum(np.square(x), axis=0, keepdims=True))
    # x = np.arange(n*m).reshape(m,n).T + 1
    X = np.row_stack((x, np.zeros((n - 1, m))))
    Z1 = circ_operation(X)

    for i in range(iterations):
        # Z_old = np.array(Z1)
        Z2 = project_operation(Z1, desired_bound)  # gamma projection
        Q = vectorize_operation3(Z2)  # should return M(g(Z2 ))
        eigs = np.linalg.eigh(Q)
        nu = eigs[0][-1]
        v = eigs[1][:, -1]
        v = v.reshape(m, N).T
        new_X = np.sqrt(nu) * v
        Z1 = circ_operation(new_X)
        if verbose:
            print(f'Iteration {i}, Convergence: {np.mean(np.square(Z2-Z1))}')

    t = new_X[:n, :]

    mu = 0
    for i in range(t.shape[1]):
        cur_filter = t[:, i]
        rest_filters = np.column_stack((t[:, :i], t[:, i + 1:]))
        for j in range(rest_filters.shape[1]):
            corrs = np.correlate(cur_filter, rest_filters[:, j], mode='full')
            mu = max(mu, np.max(corrs))

    if verbose:
        print(mu)

    if verbose:
        print(t.T.dot(t))
    print("Dictionary is")
    print(t.T.dot(t))
    t /= np.sqrt(np.sum(np.square(t), axis=0, keepdims=True))
    c=(np.abs(circ_operation(np.row_stack((t, np.zeros((n-1,m))))).dot(np.ones((2*n-1,1)))))

    c[[l*(m+1)*(2*n-1) for l in range(m)]] = 0
    mu2 = np.max(c)
    print("Normalized circular mutual coherence is {}".format(mu2))
    if verbose:
        print("Bound for pursuit algorithms success is {}".format(0.5*(1/mu2+1)))

    return t, mu2
