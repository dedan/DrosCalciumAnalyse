
import numpy as np
import scipy.sparse as sp
from scipy.ndimage import filters
import scipy.spatial.distance as dist

has_sparse = True
is_sparse = lambda A: isinstance(A, sp.spmatrix)

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
        A /= math.sqrt(alpha)
        X /= math.sqrt(alpha)
        
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


