def projection_PCA(X,m):
    """Compute the projection matrix P
        Parameters
        ----------
        X : numpy matrix of shape (n_features, n_samples)
        m : [hyperparameter] defines the number of columns of P, hence the # of considered eigenvectors
        Returns
        -------
        P : numpy matrix of shape (n_features, m)
        How it works
        ------------
        1) mu : dataset mean
        2) Xc : centered version of the dataset
        3) C : covariance matrix of shape (n_features, n_features)
        """
    mu = D.mean(axis = 1).reshape(-1,1)
    Xc = X - mu
    K = np.shape(X)[1] # number of columns of X
    C = 1/K * np.dot(Xc, Xc.T) # covariance matrix
    sigma, U = np.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m] # take the m eigenvectos of C associated to the m highest eigenvalues
    return P