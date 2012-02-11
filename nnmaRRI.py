
import numpy as np
import scipy.sparse as sp
from scipy.ndimage import filters
import scipy.spatial.distance as dist
import math

has_sparse = True
is_sparse = lambda A: isinstance(A, sp.spmatrix)

import scipy.linalg
import scipy.stats
import time
import math
import pylab as plt
import random
from scipy.spatial.distance import cdist

print 'nnmaRRI loaded'


class RRI(object):

    def frob_dist(self, Y, A, X):
        """ frobenius distance between Y and A X """
        return np.linalg.norm(Y - np.dot(A, X))
    dist = frob_dist

    def init_factors(self, Y, k, A=None, X=None, Xpart=None, shape=(64, 84)):
        """ generate start matrices U, V """
        
        print 'init started'
        m, n = Y.shape

        # sample start matrices
        if A is None: 
            A = np.random.rand(m, k)

        if X is None: 
            X = np.ones((k, n))

        # scale A, X with alpha such that || Y - alpha AX ||_fro is 
        # minimized
        AX = np.dot(A, X).flatten()
        # alpha = < Y.flatten(), AX.flatten() > / < AX.flatten(),AX.flatten() >

        alpha = np.dot(Y.flatten(), AX) / np.dot(AX, AX)
        A /= np.sqrt(np.abs(alpha) + 1E-10)
        X /= np.sqrt(np.abs(alpha) + 1E-10)
        
        return A, X

    def __call__(self, Y, k, A=None, X=None, Xpart=None, eps=1e-5,
                 maxcount=100, verbose=10, shape=(64, 84), **param):

        """ basic template for NNMA iterations """
        print 'nnma started'
        self.maxcount = maxcount
        
        YT = Y.T
        

        A, X = self.init_factors(Y, k, A, X, Xpart, shape)

        self.S = (np.diag(np.ones(X.shape[1] - 1), 1) + np.diag(np.ones(X.shape[1] - 1), -1) + np.diag(np.ones(X.shape[1] - shape[1]), shape[1]) + np.diag(np.ones(X.shape[1] - shape[1]), -shape[1]))
        for j in range(shape[0] - 1):
            pos = shape[1] * (j + 1)
            self.S[pos - 1, pos] = 0
            self.S[pos, pos - 1] = 0
        #self.S[0,1]=2
        #self.S[-1,-2]=2
        
        count = 0
        obj_old = 1e99

        param = param.copy()


        # calculate frobenius norm of Y
        nrm_Y = np.linalg.norm(Y)

        while True:

            A, X = self.update(Y, YT, A, X, count, **param)

            if np.any(np.isnan(A)) or np.any(np.isinf(A)) or \
               np.any(np.isnan(X)) or np.any(np.isinf(X)):

                if verbose: print "RESTART"
                A, X = self.init_factors(Y, k)
                count = 0
            
            count += 1

            # relative distance which is independeant to scaling of A
            obj = self.dist(Y, A, X) / nrm_Y

            delta_obj = obj - obj_old
            if verbose:
                if count % verbose == 0:
                    print "count=%6d obj=%E d_obj=%E" % (count, obj, delta_obj)

            if count >= maxcount: break 
            # delta_obj should be "almost negative" and small enough:
            if -eps < delta_obj <= 1e-12:
                break

            obj_old = obj

        
        if verbose:
            print "FINISHED:"
            print "count=%6d obj=%E d_obj=%E" % (count, obj, delta_obj)

        return A, X, obj, count, count < maxcount
    
    def update(self, Y, YT, A, X, count, **param):
        
        psi = param.get("psi", 1e-12)
        sparse_param = param.get("sparse_par", 0)
        sparse_param2 = param.get("sparse_par2", 0)
        smooth_param = param.get("smoothness", 0)
        negbase = param.get("negbase", 0)
        num_mode = A.shape[1]
        
        E = Y - np.dot(A, X)
        indlist = range(A.shape[1])
        #random.shuffle(list)
        #list = np.argsort(np.sum(X>0,1))
        for j in indlist:
            mask = np.array([True] * A.shape[1])
            mask[j] = False
            aj = A[:, j]
            xj = X[j, :]

            
            Rt = E + np.outer(aj, xj)

            xj = self.project_residuen(Rt.T, xj, aj, psi, sparse_param, smooth_param, sparse_param2=sparse_param2, X=X)
            xj /= np.max(xj) + psi
                
                                
            rect = False if (j >= (num_mode - negbase)) else True
            aj = self.project_residuen(Rt, aj, xj, psi, 0, rectify=rect)
            Rt -= np.outer(aj, xj)
            
            A[:, j] = aj
            X[j, :] = xj
            
            E = Rt

        return A, X
    
    def project_residuen(self, res, old, to_base, psi=1e-12, sparse_param=0, smoothness=0, rectify=True, sparse_param2=0, X=0):
        new_vec = np.dot(res, to_base) + psi * old 
        new_vec -= sparse_param
        
        if sparse_param2 > 0:
            norm = np.sqrt(np.sum(X ** 2, 1)).reshape((-1, 1)) + 1E-15
            occupation = np.sum(X / norm, 0)
            if np.sum(occupation[new_vec > 0] > 0):
                new_vec -= sparse_param2 * (occupation - old / (np.sqrt(np.sum(old ** 2)) + 1E-15))       
        
        if smoothness > 0:
            new_vec += smoothness * np.dot(self.S, old * np.max(new_vec))
        
        new_vec /= (np.linalg.norm(to_base) ** 2 + psi + smoothness) 
        
        if rectify:
            new_vec[new_vec < 0] = 0

        return new_vec 


def stJADE(X, **param):

    '''
    % spatiotemporal JADE (stJADE)
    %
    % decomposes the spatiotemporal data set X into
    %       X = Ss' * St
    % using double-sided joint diagonalization of fourth-order cumulants.
    % it is an extension of the Cardoso's JADE algorithm
    % usage:
    %       [Ss, St] = stJADE (X [,argID,value [...]])
    %
    % with input and output arguments ([]'s are optional):
    % X                  (matrix) data matrix X of dimension (ms x mt) representing sensor (observed) signals,
    %                            so X is a matrix gathering ms spatial observation in the rows each with mt temporal observations (in the columns).
    % [argID,           (string) Additional parameters are given as argID, value
    %   value]          (varies) pairs. See below for list of valid IDs and values.
    % Ss                (matrix) reconstructed spatial source matrix (of dimension n x ms)
    % St                (matrix) reconstructed temporal source matrix (of dimension n x mt)
    %
    % Here are the valid argument IDs and corresponding values. 
    %  'lastEig' or     (integer) Number n of components to be estimated. Default equals min(ms,mt). It is strongly
    %  'numOfIC'                 recommended to set a much smaller n here, otherwise cumulant estimates are very bad. Dimension
    %                            reduction is performed using SVD projection along eigenvectors correspoding to largest eigenvalues ('spatiotemporal PCA').
    %  'alpha'          (double) weighting parameter in [0,1] (default 0.5). The larger alpha the more temporal separation is favored.
    %  'verbose'        (string) report progress of algorithm in text format. 
    %                            Either 'on' or 'off'. Default is 'on'.
    %  'orthJD'         (string) restrict JD to orthogonal matrices only - this is much more robust than otherwise,
    %                            however it does not include the full search space (although it usually works well in practice).
    %                            Either 'on' or 'off'. Default is 'on'.
    %
    % possible extensions: use nonorthogonal diagonalization (e.g. acdc) and other source conditions (see EUSIPCO paper)
    %
    % Author : Fabian J. Theis. fabian@theis.name
    % This software is for non-commercial use only.
    %
    % ------------------------------------------------------------------------------------------------
    % REFERENCES:
    %   F.J. Theis, P. Gruber, I. Keck, A. Meyer-Baese and E.W. Lang, 
    %   'Spatiotemporal blind source separation using double-sided approximate joint diagonalization', 
    %   submitted to EUSIPCO 2005 (Antalya), 2005.
    %
    % spatiotemporal ICA (using entropies) has first been introduced by
    %   J.V. Stone, J. Porrill, N.R. Porter, and I.W. Wilkinson, 'Spatiotemporal independent component 
    %   analysis of event-related fmri data using skewed probability density functions',
    %   NeuroImage, 15(2):407-421, 2002.
    %
    % the classical ('temporal') JADE algorithm is described in
    %   J.-F. Cardoso and A. Souloumiac, 'Blind beamforming for non
    %   gaussian signals', IEE Proceedings - F, 140(6):362-370, 1993.
    %  
    % for joint diagonalization stJADE uses Cardoso's diagonalization algorithm based on iterative 
    % Given's rotations (matlab-file RJD) in the case of orthogonal searches, see
    %   J.-F. Cardoso and A. Souloumiac, 'Jacobi angles for simultaneous diagonalization',
    %   SIAM J. Mat. Anal. Appl., vol 17(1), pp. 161-164, 1995
    % and Yeredors ACDC in the general case of non-orthogonal search, see
    %   A. Yeredor, 'Non-Orthogonal Joint Diagonalization in the Least-Squares Sense with Application 
    %   in Blind Source Separation, IEEE Trans. On Signal Processing, vol. 50 no. 7 pp. 1545-1553, July 2002.
    '''


    ms, mt = X.shape

    verbose = param.get('verbose', True)
    orthJD = param.get('orthjd', True)
    n = param.get('latents', min(ms, mt))
    alpha = param.get('alpha', 0.5)
    
    if verbose:
        print('stJADE: centering data...')

        # remove the spatiotemporal mean
        Xc = X - np.mean(X, 0)
        Xc = Xc - np.mean(Xc, 1).reshape((-1, 1))

    if verbose:
        print('done\nstJADE: performing SVD for spatiotemporal PCA and dimension reduction...')


    # SVD of X and dimension reduction
    [U, D, V] = scipy.linalg.svd(Xc)
    # norm(Xc - U * D * V')
    Dx = D
    D = np.diag(D)
    D = D[:n, :n]
    U = U[:, :n]
    V = V.T[:, :n]
    #norm(Xc-U*D*V')

    if verbose:
        print('done\n')
    if min(ms, mt) > n:
        print 'retained percentage of variance: ', np.sum(Dx[:n]) / np.sum(Dx) * 100
    else:
        print('stJADE: warning - no dimension reduction has been performed. It is strongly recommended to set a much smaller n using parameter ''latents'', otherwise cumulant estimates are very bad.')
    

    #########################################################################################
    # Estimation of the multidimensional space delayed covariance matrices
    #########################################################################################
    nummat = n * (n + 1) / 2;
    if verbose:
        print('stJADE: calculating 2 x %i contracted cumulant matrices...', nummat, type(nummat))

    M = np.zeros((n, n, 2 * nummat))
    temp = np.dot(D ** 0.5, V.T)
    M[:, :, :nummat] = jadeCummulantMatrices(temp)
    M[:, :, nummat:] = jadeCummulantMatrices(np.dot(D ** 1.5, U.T))
    for i in range(nummat, 2 * nummat):
        M[:, :, i] = np.linalg.matrix_power(M[:, :, i] , -1)

    # normalization within the groups in order to allow for comparisons using alpha
    M[:, :, :nummat] = alpha * M[:, :, :nummat] / np.mean(np.sqrt(np.sum(M[:, :, :nummat] ** 2)));
    M[:, :, nummat:] = (1 - alpha) * M[:, :, nummat:] / np.mean(np.sqrt(np.sum(M[:, :, nummat :] ** 2)));
    # also possible: normalization by mean determinant

    #########################################################################################
    # Joint diagonalization
    #########################################################################################
 
    if verbose:
        print('done\nstJADE: performing approximate joint diagonalization of %i (%ix%i)-matrices ', 2 * nummat, n, n);

    if orthJD:
        if verbose:
            print('using orthogonal JD...')
        W = rjd(np.hstack([M[:, :, i] for i in range(M.shape[2])])).T #M.reshape((n, -1))).T # % orthogonal approximate JD
    else: #careful: acdc doesn't really like noninvertible matrices       
        print 'not yet implemnted'
        #if verbose:
        #    print('using nonorthogonal JD...')
        #    W = acdc_sym(M) ^ (-1) # non - orthogonal approximate JD

    if verbose:
        print('done\n')

    #########################################################################################
    # Output
    #########################################################################################

    St = np.dot(np.dot(W , np.diag(np.diag(D) ** -0.5)) , V.T)
    Ss = np.dot(np.dot(U , np.diag(np.diag(D) ** 1.5)), np.linalg.matrix_power(W , -1)).T
       
    # add transformed means
    means = np.mean(X, 0)
    meant = np.mean(X, 1)
    
    pSs = scipy.linalg.pinv(Ss).T
    pSt = scipy.linalg.pinv(St).T

    Smeant = np.dot(pSs, meant)
    Smeans = np.dot(pSt, means)
    
    St = St + Smeant.reshape((-1, 1))
    Ss = Ss + Smeans.reshape((-1, 1))

    return Ss, St

def jadeCummulantMatrices(X, useUnitCov=False):

    # calcs the n(n+1)/2 cum matrices used in JADE, see C&A book, page 173 (pdf:205), C.1
    # does not need whitened data X (n x T)
    # returns M as n x n x (n(n+1)/2) array
    # set useUnitCov to 1 if you want to use I for covariance (e.g. if data has been prewhitened)

    n, T = X.shape
    k = 0
    M = np.zeros((n, n, n * (n + 1) / 2))
    scale = np.ones((n, 1)) / T

    if useUnitCov:
        R = np.eye(n)
    else:
        R = np.cov(X) # covariance    


    for p in range(n):
        # case q = p
        C = np.dot(np.outer(scale, (X[p, :] * X[p, :])) * X , X.T)
        E = np.zeros((n, n))
        E[p, p] = 1
        M[:, :, k] = C - np.dot(R, np.dot(E, R)) - np.trace(np.dot(E, R)) * R - np.dot(np.dot(R, E.T), R)
        k = k + 1;
        # case q < p
        for q in range(p):
            
            C = np.dot(np.outer(scale , X[p, :] * X[q, :]) * X , X.T) * np.sqrt(2)
            E = np.zeros((n, n))
            E[p, q] = 1 / np.sqrt(2)
            E[q, p] = E[p, q]
            M[:, :, k] = C - np.dot(R, np.dot(E, R)) - np.trace(np.dot(E, R)) * R - np.dot(np.dot(R, E.T), R)
            k = k + 1;
            
    return M
    
def rjd(A, threshold=np.finfo(float).eps):
    '''
    %***************************************
    % joint diagonalization (possibly
    % approximate) of REAL matrices.
    %***************************************
    % This function minimizes a joint diagonality criterion
    % through n matrices of size m by m.
    %
    % Input :
    % * the  n by nm matrix A is the concatenation of m matrices
    %   with size n by n. We denote A = [ A1 A2 .... An ]
    % * threshold is an optional small number (typically = 1.0e-8 see below).
    %
    % Output :
    % * V is a n by n orthogonal matrix.
    % * qDs is the concatenation of (quasi)diagonal n by n matrices:
    %   qDs = [ D1 D2 ... Dn ] where A1 = V*D1*V' ,..., An =V*Dn*V'.
    %
    % The algorithm finds an orthogonal matrix V
    % such that the matrices D1,...,Dn  are as diagonal as possible,
    % providing a kind of `average eigen-structure' shared
    % by the matrices A1 ,..., An.
    % If the matrices A1,...,An do have an exact common eigen-structure
    % ie a common othonormal set eigenvectors, then the algorithm finds it.
    % The eigenvectors THEN are the column vectors of V
    % and D1, ...,Dn are diagonal matrices.
    % 
    % The algorithm implements a properly extended Jacobi algorithm.
    % The algorithm stops when all the Givens rotations in a sweep
    % have sines smaller than 'threshold'.
    % In many applications, the notion of approximate joint diagonalization
    % is ad hoc and very small values of threshold do not make sense
    % because the diagonality criterion itself is ad hoc.
    % Hence, it is often not necessary to push the accuracy of
    % the rotation matrix V to the machine precision.
    % It is defaulted here to the square root of the machine precision.
    % 
    %
    % Author : Jean-Francois Cardoso. cardoso@sig.enst.fr
    % This software is for non commercial use only.
    % It is freeware but not in the public domain.
    % A version for the complex case is available
    % upon request at cardoso@sig.enst.fr
    %-----------------------------------------------------
    % Two References:
    %
    % The algorithm is explained in:
    %
    %@article{SC-siam,
    %   HTML =    "ftp://sig.enst.fr/pub/jfc/Papers/siam_note.ps.gz",
    %   author = "Jean-Fran\c{c}ois Cardoso and Antoine Souloumiac",
    %   journal = "{SIAM} J. Mat. Anal. Appl.",
    %   title = "Jacobi angles for simultaneous diagonalization",
    %   pages = "161--164",
    %   volume = "17",
    %   number = "1",
    %   month = jan,
    %   year = {1995}}
    %
    %  The perturbation analysis is described in
    %
    %@techreport{PertDJ,
    %   author = "{J.F. Cardoso}",
    %   HTML =    "ftp://sig.enst.fr/pub/jfc/Papers/joint_diag_pert_an.ps",
    %   institution = "T\'{e}l\'{e}com {P}aris",
    %   title = "Perturbation of joint diagonalizers. Ref\# 94D027",
    %   year = "1994" }
    %
    %
    %
    '''
    
    m, nm = A.shape
    V = np.eye(m)

    encore = True
    while encore:
        encore = False
        print 'next iter'
        for p in range(m):
            for q in range(p, m):
                ###computation of Givens rotations
                g = np.vstack((A[p, p:nm:m] - A[q, q:nm:m] , A[p, q:nm:m] + A[q, p:nm:m]))
                g = np.dot(g , g.T)
                ton = g[0, 0] - g[1, 1]
                toff = g[0, 1] + g[1, 0]
                theta = 0.5 * math.atan2(toff , ton + np.sqrt(ton * ton + toff * toff))
                c = math.cos(theta)
                s = math.sin(theta)
                encore = np.logical_or(encore, (abs(s) > threshold))
                # update of the A and V matrices 
                if (abs(s) > threshold):
                    Mp = A[:, p:nm:m]
                    Mq = A[:, q:nm:m]
                    A[:, p:nm:m] = c * Mp + s * Mq
                    A[:, q:nm:m] = c * Mq - s * Mp
                    rowp = A[p, :]
                    rowq = A[q, :]
                    A[p, :] = c * rowp + s * rowq
                    A[q, :] = c * rowq - s * rowp
                    temp = V[:, p]
                    V[:, p] = c * V[:, p] + s * V[:, q]
                    V[:, q] = c * V[:, q] - s * temp
    qDs = A 
    return V #, qDs

'''
def testjade()

    n=4;
    # make some toy images
    Ss=zeros(64,64,n);
    for i in range(2*n) 
        a=np.ceil(rand*40)
        b=np.ceil(rand*40)
        da=np.ceil(rand*20)
        db=np.ceil(rand*20);
    Ss(a:a+da,b:b+db,mod(i,n)+1)=rand(da+1,db+1); 


% plot them
figure, showdata(Ss,'spatial sources')

% wait
pause

% make some toy time series
St(1,:)=mod(ceil((1:100)/10),2);
St(2,:)=randn(1,100);
St(3,:)=sin((1:100)/0.02+2);
St(4,:)=sin((1:100)/0.045+1);

% plot them
figure, showdata(St, 'temporal sources')

% wait
pause

% mix them
X=mixim(St',Ss);

% plot the first entries of X
figure, showdata(X(:,:,1:10),'first 10 mixture images')

% wait
pause

% perform stSOBI
[Ssest,Stest]=stSOBI(X,'lastEig',n,'orthJD','on');

% plot results
figure, showdata(Stest,'recovered temporal sources using stSOBI')
figure, showdata(Ssest,'recovered spatial sources using stSOBI')

% give some indication for separation performance 
% -> the following matrix should be close to a permutation matrix
Stest*pinv(St)

% stSOBI performance is not excellent because of random structure in images
% actually autocovs are rather degenerate... (in contrast to real images) 

% wait
pause

close all
[Ssest,Stest]=stJADE(reshape(X,64^2,100),'lastEig',n,'orthJD','on');
Ssest=mixim(pinv(Stest'),X);

% plot results
figure, showdata(Stest,'recovered temporal sources using stJADE')
figure, showdata(Ssest,'recovered spatial sources using stJADE')

% give some indication for separation performance 
% -> the following matrix should be close to a permutation matrix
Stest*pinv(St)

echo off
'''
