def logpdf_GAU_ND_1sample(x,mu,C):
    """This function computes the Gaussian probability density for a given sample x.
        M: number of features
        Parameters
        ----------
        x: sample | numpy array of shape (M, 1) 
        mu: mean  | numpy array of shape (M, 1)
        C: covariance matrix | numpy array of shape (M,M)
        Returns
        -------
        N(x|mu,C) -> gaussian probability density associated to x given mu and C (it is a scalar)
        """
    M = x.shape[0]
    xc = x - mu # x centered
    invC = np.linalg.inv(C) # C^-1
    _,log_abs_detC = np.linalg.slogdet(C) # log(abs(C))
    return -M/2 * np.log(2*np.pi) - 1/2 * log_abs_detC - 1/2 * np.dot(np.dot(xc.T,invC),xc)

def logpdf_GAU_ND(x,mu,C):
    """This function computes the Gaussian probability density for a given set of samples x=(x1,x2,...xn).
        M: number of features
        N: number of samples
        Parameters
        ----------
        x: matrix of samples | numpy array of shape (M, N) 
        mu: mean  | numpy array of shape (M, 1)
        C: covariance matrix | numpy array of shape (M,M)
        Returns
        -------
        y=(N(x1|mu,C),N(x2|mu,C),...,N(xn|mu,C) -> gaussian probability densities associated to each sample xi given mu and C | numpy array of shape (N,)
        """
    M = x.shape[0]
    N = x.shape[1]
    y = np.zeros(N) # array of N scalar elements
    for i in range(N):
        density_xi = logpdf_GAU_ND_1sample(x[:,i:i+1],mu,C)
        y[i] = density_xi
    return y

def loglikelihood(X):
    """This function computes the loglikelihood function value
        mu_ML: mean  | numpy array of shape (M, 1)
        C_ML: covariance matrix | numpy array of shape (M,M)
        Parameters
        ----------
        X: matrix of samples | numpy array of shape (M, N) 
        Returns
        -------
        loglikelihood
        """
    N = X.shape[1]
    mu_ML = np.mean(X, axis=1).reshape(-1,1)
    Xc = X - mu_ML
    C_ML = 1/N * np.dot(Xc,Xc.T) # covariance matrix
    return sum(logpdf_GAU_ND(X,mu_ML,C_ML))