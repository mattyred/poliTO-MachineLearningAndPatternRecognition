def compute_mu_matrix(D,labels):
    """Compute the matrix that has for each column the average value for each class label (a representative sample). Given D[n_features,n_samples] we obtain mu_matrix[n_features,n_classes]
        Parameters
        ----------
        D : numpy matrix with the dataset | shape: (n_features, n_samples)
        labels : numpy array with labels for each class | shape: (n_samples)
        Returns
        -------
        mu_matrix : numpy matrix | shape: (n_features, n_classes)
        """
    n_features = D.shape[0]
    K = len(set(labels)) # number of classes
    mu_matrix = np.zeros(shape=(n_features,K))
    mu_dataset = D.mean(axis=1).reshape(-1,1)
    for i in set(labels):
        samples_class_i = D[:,labels == i]
        samples_class_i_mean = np.mean(samples_class_i, axis=1).reshape(-1,1)
        mu_matrix[:,i:i+1] = samples_class_i_mean
    mu_matrix -= mu_dataset
    return mu_matrix

def compute_SB(D,labels):
    """Compute SB matrix
        Parameters
        ----------
        D : numpy matrix with the dataset | shape: (n_features, n_samples)
        labels : numpy array with labels for each class | shape: (n_samples)
        Returns
        -------
        SB : numpy matrix | shape: (n_features, n_features)
        """
    K = len(set(labels)) # number of classes
    N = D.shape[1] # number of samples
    SB = 0
    nc = np.array([np.sum(labels==i) for i in set(labels)]) # number of samples for each class label
    mu_matrix = compute_mu_matrix(D,labels)
    for i in range(K):
        SB += nc[i] * np.dot(mu_matrix[:,i:i+1], mu_matrix[:,i:i+1].T)
    SB /= N
    return SB

def compute_SB(D,labels):
    """Compute SW matrix
        Parameters
        ----------
        D : numpy matrix with the dataset | shape: (n_features, n_samples)
        labels : numpy array with labels for each class | shape: (n_samples)
        Returns
        -------
        SW : numpy matrix | shape: (n_features, n_features)
        """
    SWc = 0
    K = len(set(labels)) # number of classes
    N = D.shape[1] # number of samples
    SW = 0
    nc = np.array([np.sum(labels==i) for i in set(labels)]) # number of samples for each class label
    for i in range(K):
        samples_class_i = D[:,labels == i]
        samples_class_i_mean = np.mean(samples_class_i, axis=1).reshape(-1,1)
        samples_class_i_centered = samples_class_i - sample_class_i_mean # center the samples of class i by subtracting the mean of the class
        SWc = 1/nc[i] * np.dot(samples_class_i_centered, samples_class_i_centered.T)
        SW += nc[i] * SWc # outer summary
    SW /= N
    return SW

def compute_W(D,labels,m=-1,norm=False):
    """Compute W matrix of LDA directions
        Parameters
        ----------
        D : numpy matrix with the dataset | shape: (n_features, n_samples)
        labels : numpy array with labels for each class | shape: (n_samples)
        Returns
        -------
        W : numpy matrix 
        """
    SB = compute_SB(D,labels)
    SW = compute_SW(D,labels)
    if m == -1: #default value
        m = K-1
    sigma, U = scipy.linalg.eigh(SB, SW)
    W = U[:, ::-1][:, 0:m]
    
    if norm == True:
        UW, _, _ = np.linalg.svd(W)
        Wnorm = UW[:, 0:m]
        return Wnorm # W with orthogonal columns
    else:
        return W # W with no orhogonal columns
    
def compute_LDA_features(D,labels,m=-1,norm=False):
    """Compute the new features with LDA
        Parameters
        ----------
        D : numpy matrix with the dataset | shape: (n_features, n_samples)
        labels : numpy array with labels for each class | shape: (n_samples)
        Returns
        -------
        y : numpy matrix 
        """
    W = compute_W(D,labels,m,norm)
    y = np.dot(W.T,D)
    return y