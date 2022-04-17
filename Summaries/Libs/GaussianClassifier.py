def split_db_2to1(D, L, seed=0):
        """Split the dataset D in two subsets, one for training and one for testing
        Parameters
        ----------
        D: numpy array of shape (n_features, n_samples)
        L: numpy array of shape (n_classes, n_samples)
        Returns
        -------
        (DTR, LTR), (DTE, LTE) : 'TR' means training and 'TE' means testing
        """
    nTrain = int(D.shape[1]*2.0/3.0) # 2/3 of the dataset D are used for training, 1/3 for validation
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1]) # take a random array of 150 elements, each element is 0<x<=149 (np.arange(150))
    idxTrain = idx[0:nTrain] # first 100 are indices of training samples 
    idxTest = idx[nTrain:] # remaining 50 are indices of validation samples
    DTR = D[:, idxTrain] # D for training
    DTE = D[:, idxTest] # D for validation
    LTR = L[idxTrain] # L for training
    LTE = L[idxTest] # L for validation
    return (DTR, LTR), (DTE, LTE)

def train_model(DTR, LTR):
     """Trains the Generative Gaussian Classifier model by fitting a normal distribution (according to the maximum likelihood estimation) to the training set of data.
        Parameters
        ----------
        DTR: numpy array of shape (n_features, n_samples) -- training sample
        LTR: numpy array of shape (n_classes, n_samples) -- training labels
        Returns
        -------
        (mu_classes,cov_classes): tuple of python lists containing the empirical mean and covariance matrix for each class
        """
    mu_classes = [] # list of empiracal mean for each class
    cov_classes = [] # list of covariance matrix for each class
    for i in set(LTR):
        DTR_class_i = DTR[:,LTR==i]
        N_class_i = DTR_class_i.shape[1]
        mu_class_i = DTR_class_i.mean(axis=1).reshape(-1,1)
        cov_class_i = 1/N_class_i * np.dot(DTR_class_i-mu_class_i, (DTR_class_i-mu_class_i).T)
        mu_classes.append(mu_class_i)
        cov_classes.append(cov_class_i)
    return (mu_classes, cov_classes)

def logpdf_GAU_ND_1sample(x,mu,C):
         """Find details in MVG.py
            """
    M = x.shape[0] # num of features of sample x
    mu = mu.reshape(M,1) # mean of the sample
    xc = x - mu # x centered
    invC = np.linalg.inv(C)
    _,log_abs_detC = np.linalg.slogdet(C)
    return -M/2 * np.log(2*np.pi) - 1/2 * log_abs_detC - 1/2 * np.dot(np.dot(xc.T,invC),xc)

def compute_post_probabilities(DTE, LTE, mu_classes, cov_classes):
         """ Compute the class conditional probabilities for each class, for each sample. It is Fx|c(xt|0),Fx|c(xt|1),Fx|c(xt|2) for a 3-labels classification problem
        Parameters
        ----------
        DTE: numpy array of shape (n_features, n_samples) -- testing samples
        LTE: numpy array of shape (n_classes, n_samples) -- testing labels
        Returns
        -------
        SPost: numpy array of shape (n_features, n_samples) containg in [i,j] the probability of sample j to belong to class i
        -------
        1)store class-conditional probabilities in a score matrix S
          S[i, j] should be the class conditional probability for sample j given class i
        2)we assume a prior probabilty for each class => 1/Nclases = P(c) for every c
        3)compute joint density matrix SJoint => Fx,c(xt,c) = Fx|c(xt|c) * P(c)
        4)compute SMarginal by summing in each class the values of SJoint
        5)compute SPost 
        """
    S = np.zeros(shape=(LTE.shape[0],DTE.shape[1]))
    for i in range(DTE.shape[1]):
        xt = DTE[:,i:i+1] # test sample xt
        # now compute the probability density related to each class label for the sample xt
        score = np.zeros(shape=(3,1))
        for j in set(LTE):
            mu = mu_classes[j]
            C = cov_classes[j]
            score[j,:] = np.exp(logpdf_GAU_ND_1sample(xt,mu,C))
        S[:,i:i+1] = score
        
    prior_prob = 1 / LTE.shape[0]
    SJoint = S * prior_prob
    SMarginal = SJoint.sum(0).reshape(-1,1)
    # compute class posterior probabilities SPost = SJoint / SMarginal
    SPost = np.zeros((LTE.shape[0],LTE.shape[1]))
    for c in range(LTE.shape[0]):
        SJoint_c = SJoint[c,:].reshape(-1,1)
        SPost_c = (SJoint_c / SMarginal).reshape(1,-1)
        SPost[c,:] = SPost_c
    return SPost

def compute_post_probabilities_log(DTE, LTE, mu_classes, cov_classes):
    ############ EQUIVALENT TO compute_post_probabilities ###############
    S = np.zeros(shape=(LTE.shape[0],DTE.shape[1]))
    for i in range(DTE.shape[1]):
        xt = DTE[:,i:i+1]
        score = np.zeros(shape=(3,1))
        for j in set(LTE):
            mu = mu_classes[j]
            C = cov_classes[j]
            score[j,:] = np.exp(logpdf_GAU_ND_1sample(xt,mu,C))
        S[:,i:i+1] = score
        
    prior_prob = 1 / LTE.shape[0]
    SJoint = S * prior_prob
    SMarginal = SJoint.sum(0).reshape(-1,1)
    SPost = np.zeros((LTE.shape[0],LTE.shape[1]))
    for c in range(LTE.shape[0]):
        SJoint_c = SJoint[c,:].reshape(-1,1)
        SPost_c = (SJoint_c / SMarginal).reshape(1,-1)
        SPost[c,:] = SPost_c
    ######################################################################
    logSJoint = np.log(SJoint) + np.log(1/3)
    logSMarginal = scipy.special.logsumexp(logSJoint, axis=0).reshape(1,-1)
    log_SPost = logSJoint - logSMarginal  
    SPost_ = np.exp(log_SPost)
    return SPost

def predict_labels(SPost ,LTE):
        """ Find predicted class labels. Each sample is assigned to the class for which it has the highest probability in the matrix SPost
        Parameters
        ----------
        SPost: numpy array of shape (n_features, n_samples) containg in [i,j] the probability of sample j to belong to class i
        Returns
        -------
        (predicted_labels, accuracy, error_rate)
        """
    predicted_labels = np.argmax(SPost,axis=0)
    corrected_assigned_labels = LTE==predicted_labels
    acc = sum(corrected_assigned_labels) / len(LTE)
    err = 1-acc
    return (predicted_labels, acc, err)
    
    