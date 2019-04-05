########################################################
#     Variational EM algorithm for source separation
########################################################

# '''
# I : number of receivers
# J : number of sources
# T : duration of the received signals
# Ls : duration of source signals
# La : length of mixing filters
# F : spectral resolution of MDCT for each source j [J x 1]
# N : time resolution of MDCT for each source j [J x 1]
# M : window length of MDCT for each source j [J x 1]
# H : hop size for MDCT for each source j [J x 1]
# K : NMF rank for each source j [J x 1]
# fs : sample rate
# x : observations matrix [I x T]
# alpha_u : Student's T shape parameter for mixing filters [1]
# alpha_v : Student's T shape parameter for sources [1]
# sigma2_n : noise variance on each receivers [I x 1]
# W : spectral template matrix for each source [J x Fj x Kj]
# H : activation matrix for each source [J x Kj x Nj]
#
# '''

import numpy as np
from scipy.stats import invgamma
from utility_functions import *
from scipy.sparse.linalg import cg
from scipy.linalg import toeplitz
from scipy.special import psi, gammaln
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat



class VEM:

    def __init__(self, fs, x, alpha_u, alpha_v, sigma2_n, r2, W, H, wlen):

        self.eps = np.finfo(float).eps

        self.fs = fs; # sample_rate
        self.x = x;
        savemat('/cal/exterieurs/atiam6576/Téléchargements/code-SSMM-MASS-2017/data/x.mat', mdict={'x': self.x})

        self.alpha_u = alpha_u;
        self.alpha_v = alpha_v;
        self.sigma2_n = sigma2_n;
        self.r2 = r2;
        self.H = H;
        self.W = W;

        self.T = int(self.x.shape[0]);
        self.I = int(self.x.shape[1]);
        self.J = int(len(self.W));
        self.La = int(self.alpha_u.shape[0]);
        self.Ls = int(self.T - self.La + 1);
        self.wlen = wlen;
        self.F = [None]*self.J;
        self.N = [None]*self.J;

        self.sig_source = np.empty((self.J, self.Ls));
        self.phiCorr = [None] * self.J;
        self.s_im = np.empty((self.I,self.J,self.T));
        self.expectError = np.empty(self.I)
        self.varFreeEnergy = 0;

        for j in range(self.J):
            self.F[j] = int(self.wlen[j]/2);
            self.N[j] = int(np.round(self.Ls/self.F[j]) - 1);

        self.lambda2 = [None]*self.J;
        for j in range(self.J):
            self.lambda2[j] = np.dot(self.W[j],self.H[j]);

        #savemat('/cal/exterieurs/atiam6576/Téléchargements/code-SSMM-MASS-2017/data/lambda2.mat', mdict={'lambda2': self.lambda2})


        #### Initialization of variational parameters
        # V
        self.nu_v = (self.alpha_v + 1)/2;
        self.beta = [];
        for j in range(self.J):
            beta_j = self.alpha_v/2*np.ones((self.F[j],self.N[j]));
            self.beta.append(beta_j);

        # S
        self.gamma = [];
        self.s_hat = [];
        # for j in range(self.J):
        #     gamma_j = invgamma.rvs(self.nu_v, scale=self.beta[j])*self.lambda2[j];
        #     self.gamma.append(gamma_j);
        #     s_hat_j = np.random.randn(self.F[j],self.N[j])*np.sqrt(gamma_j);
        #     self.s_hat.append(s_hat_j);

        # gamma = loadmat('data/gamma.mat');
        # gamma = gamma['gamma'];
        for j in range(self.J):
            # self.gamma.append(gamma[j][0])
            self.gamma.append(0.5*np.ones((self.F[j],self.N[j])))

        # s_hat = loadmat('data/s_hat.mat');
        # s_hat = s_hat['s_hat'];
        for j in range(self.J):
            # self.s_hat.append(s_hat[j][0])
            self.s_hat.append(0.5*np.ones((self.F[j],self.N[j])))

        # U
        self.nu_u = (self.alpha_u + 1)/2;
        d_temp = np.repeat((self.alpha_u/2).reshape((1,1,-1)),self.I,axis=0)
        self.d = np.repeat(d_temp, self.J, axis=1);

        # A
        # self.a_hat = np.zeros((self.I,self.J,self.La));
        # self.rho = np.zeros((self.I,self.J,self.La));
        # for i in range(self.I):
        #     for j in range(self.J):
        #         rho_ij = invgamma.rvs(self.nu_u[0], scale=self.d[i,j,:])*r2
        #         self.rho[i,j,:] = rho_ij;
        #         self.a_hat[i,j] = np.random.randn(self.La)*np.sqrt(rho_ij)

        # rho = loadmat('data/rho.mat')
        # self.rho = rho['rho']
        # a_hat = loadmat('data/a_hat.mat')
        # self.a_hat = a_hat['a_hat']

        self.rho = 0.5*np.ones((self.I, self.J, self.La))
        self.a_hat = 0.5*np.ones((self.I, self.J, self.La))

        self.G = [np.empty((self.F[0], self.wlen[0] + self.La -1, self.I))] *self.J

        self.compute_G_simon()
        self.computePhiCorr();
        #savemat('/cal/exterieurs/atiam6576/Téléchargements/code-SSMM-MASS-2017/data/phiCorr.mat', mdict={'phiCorr': self.phiCorr})

        self.computeSourceSignal();
        # print(self.sig_source)
        savemat('/cal/exterieurs/atiam6576/Téléchargements/code-SSMM-MASS-2017/data/sig_s.mat', mdict={'sig_s': self.sig_source})
        # #savemat('/cal/exterieurs/atiam6576/Téléchargements/code-SSMM-MASS-2017/data/s_hat.mat', mdict={'s_hat': self.s_hat})
        # W_init = loadmat('data/s.mat');
        # s = W_init['s']
        # self.sig_source = s.T
        # print(self.sig_source)
        self.computeSourceImageSignal();
        # savemat('/cal/exterieurs/atiam6576/Téléchargements/code-SSMM-MASS-2017/data/s_im.mat', mdict={'s_im': self.s_im})
        # W_init = loadmat('data/sim.mat');
        # sim = W_init['sim'];
        # self.s_im = sim.T
        # print(self.s_im)
        # print(self.s_im.shape)
    ## E-V step
    def updateV(self):
        for j in range(self.J):
            self.beta[j] = self.alpha_v/2 + (self.s_hat[j]**2 + self.gamma[j])/(2*self.lambda2[j] + self.eps);

    ## E-U step
    def updateU(self):

        self.nu_u = (self.alpha_u + 1)/2

        for i in range(self.I):
            for j in range(self.J):
                self.d[i,j,:] = (self.alpha_u/2).flatten() + (self.a_hat[i,j,:]**2 + self.rho[i,j,:])/(2*self.r2 + self.eps)

    ## E-S step
    def updateS(self):

        ##  Update gamma
        inv_v_post = [None] * self.J
        for j in range(self.J):
            inv_v_post[j] = self.nu_v/(self.beta[j] + self.eps)
            norm2_g = np.sum(self.G[j]**2, axis=1) # F x I
            sum_rho = np.sum(self.rho[:,j,:], axis=1) # I
            sum_i = np.sum(1/(self.sigma2_n.flatten() + self.eps)*(sum_rho + norm2_g), axis=1).reshape(-1,1) # F x 1
            self.gamma[j] = 1/(inv_v_post[j]/(self.lambda2[j] + self.eps) + sum_i)

        ## Update s with pcg algorithm
        # compute gradient
        epsilon = self.x - np.sum(self.s_im, axis=1).T
        inv_sigma2_sum_rho = np.sum(1/self.sigma2_n.reshape(-1,1) * np.sum(self.rho, axis=2),axis=0)

        frameInd = [None] * self.J
        sum_GX = [None] * self.J
        grad = [None] * self.J
        for j in range(self.J):
            hop = self.wlen[j]/2
            frameInd[j] = np.zeros((self.wlen[j] + self.La - 1, self.N[j]), dtype=int)
            for n in range(self.N[j]):
                frameInd[j][:,n] = n*hop + np.arange(self.wlen[j] + self.La - 1)

            epsilonMat = np.zeros((self.I, self.wlen[j] + self.La - 1, self.N[j]))

            for i in range(self.I):
                epsilon_i = epsilon[:,i]/self.sigma2_n[i]
                epsilonMat[i,:,:] = epsilon_i[frameInd[j]]

            grad[j] = self.s_hat[j]*(inv_v_post[j]/(self.lambda2[j] + self.eps) + inv_sigma2_sum_rho[j]) \
                - np.sum(np.matmul(self.getGj(j), epsilonMat), axis=0)

        # w
        w = [None] * self.J
        for j in range(self.J):
            w[j] = self.gamma[j]*grad[j]

        kappa = [None] * self.J
        gradP = [None] * self.J

        niter = 10
        for iter in range(niter):

            #compute kappa
            w_mdct = np.zeros((self.Ls, self.J))
            for j in range(self.J):
                w_mdct[:,j] = imdct(w[j], self.Ls)
            w_filt = self.computeWFilt(w_mdct);
            w_filt = np.sum(w_filt, axis=1)

            for j in range(self.J):

                w_filtMat = np.zeros((self.I, self.wlen[j] + self.La -1, self.N[j]))
                for i in range(self.I):
                    w_filt_i = w_filt[:,i]/self.sigma2_n[i]
                    w_filtMat[i,:,:] = w_filt_i[frameInd[j]]

                kappa[j] = w[j]*(inv_v_post[j]/(self.lambda2[j] + self.eps) + inv_sigma2_sum_rho[j]) + np.sum(np.matmul(self.getGj(j),w_filtMat), axis=0)

            # comput mu
            mu = np.matmul(np.asarray(w).flatten().T,np.asarray(grad).flatten())/np.matmul(np.asarray(w).flatten().T,np.asarray(kappa).flatten())
            if mu<0: mu = self.eps


            # update paramaters
            for j in range(self.J):
                self.s_hat[j] = self.s_hat[j] - mu*w[j]

            # update sources
            self.computeSourceSignal()
            self.computeSourceImageSignal()

            if iter == niter - 1:
                break

            epsilon = self.x - np.sum(self.s_im, axis=1).T
            for j in range(self.J):

                epsilonMat = np.zeros((self.I, self.wlen[j] + self.La - 1, self.N[j]))
                for i in range(self.I):
                    epsilon_i = epsilon[:,i]/self.sigma2_n[i]
                    epsilonMat[i,:,:] = epsilon_i[frameInd[j]]

                grad[j] = self.s_hat[j]*(inv_v_post[j]/(self.lambda2[j] + self.eps) + inv_sigma2_sum_rho[j]) - np.sum(np.matmul(self.getGj(j), epsilonMat), axis=0)

            # preconditionning
            for j in range(self.J):
                gradP[j] = self.gamma[j]*grad[j]
            # compute alpha
            alpha_pcg = - np.matmul(np.asarray(kappa).flatten().T,np.asarray(gradP).flatten())/np.matmul(np.asarray(w).flatten().T,np.asarray(kappa).flatten())
            if alpha_pcg<0: alpha_pcg = self.eps

            # compute
            for j in range(self.J):
                w[j] = gradP[j] + alpha_pcg*w[j]

    ## E-A step
    def updateA(self):

        # update rho
        for j in range(self.J):
            sum_gamma = np.sum(np.sum(self.gamma[j]));
            norm2_s = np.sum(self.sig_source[j,:]**2);
            sigma_term = (sum_gamma + norm2_s)*1/self.sigma2_n
            nu_term = self.nu_u.flatten()/(self.d[:,j,:]*self.r2)
            self.rho[:,j,:] = 1/(nu_term + sigma_term.reshape(-1,1))

        ## compute auxiliary functions
        # autoCorr de S_hat
        r_ss = [None]*self.J
        for j in range(self.J):
            r_ss[j] = xcorr(self.sig_source[j,:], self.sig_source[j,:], self.La)

        # sum_gamma_r_phiphi et toeplitz matrix
        gam_r = np.zeros((self.La,self.J))
        toeplitz_mat = np.empty((self.I,self.J,self.La,self.La))
        toep_diag = np.zeros((self.I,self.J,self.La))
        for j in range(self.J):
            gamma_jfn = self.gamma[j]
            for t in range(self.La):
                phiCorr_t = self.phiCorr[j][:,t].reshape(-1,1)
                phiCorr_t = np.tile(phiCorr_t, [self.N[j],1])
                gam_r[t,j] = np.dot(phiCorr_t.T, gamma_jfn.flatten())

            for i in range(self.I):
                toep_diag[i,j,:] = 1/self.sigma2_n[i]*(r_ss[j] + gam_r[:,j])
                toeplitz_mat[i,j,:,:] = toeplitz(toep_diag[i,j,:])

        lambda_a = np.empty((self.I,self.J,self.La,self.La))
        epsilon_ij = np.empty((self.I, self.J, self.T))
        r_se = np.empty((self.I, self.J, self.La))
        r_se_sigma = np.empty(r_se.shape)
        for j in range(self.J):
            for i in range(self.I):

                # Lambda_a
                tmp = self.nu_u.flatten()/(self.d[i,j,:]*self.r2).flatten()
                diag = np.diag(tmp)
                lambda_a[i,j,:,:] = diag + toeplitz_mat[i,j,:,:]

                # epsilon_ij et r_se/sigma2_n
                y_ij_index = np.delete(np.arange(self.J),j)
                sum_y_ij = np.sum(self.s_im[i,y_ij_index,:], axis=0)

                epsilon_ij[i,j,:] = self.x[:,i] - sum_y_ij
                tmp = np.flip(xcorr(self.sig_source[j,:].reshape(-1,1), epsilon_ij[i,j,:].reshape(-1,1), self.La, zeroPad=True))

                r_se[i,j,:] = 1/self.sigma2_n[i]*tmp.flatten()

                grad_ij = np.matmul(lambda_a[i,j,:,:],self.a_hat[i,j,:]) - r_se[i,j,:]

                w_ij = self.rho[i,j,:]*grad_ij

                niter = 10

                for iter in range(niter):

                    kappa_ij = np.matmul(lambda_a[i,j,:,:],w_ij)

                    mu = np.dot(w_ij.T,grad_ij)/np.dot(w_ij.T,kappa_ij)
                    if mu<0:
                        mu = self.eps

                    self.a_hat[i,j,:] = self.a_hat[i,j,:] - mu*w_ij

                    if iter == niter:
                        break

                    grad_ij = np.matmul(lambda_a[i,j,:,:],self.a_hat[i,j,:]) - r_se[i,j,:]

                    gradP_ij = self.rho[i,j,:]*grad_ij

                    alpha_pcg = - np.dot(kappa_ij.T,gradP_ij)/np.dot(w_ij.T,kappa_ij)
                    if alpha_pcg<0:
                        alpha_pcg = self.eps

                    w_ij = gradP_ij + alpha_pcg*w_ij

                # precondA_hat = np.diag(1/np.diag(lambda_a[i,j,:,:]))
                #
                # ## Compute preconditionned conjugate gradient descent
                # self.a_hat[i,j,:],conv = cg(lambda_a[i,j,:,:], r_se[i,j,:].reshape(-1,1), M=precondA_hat, maxiter=10)
                # print(self.a_hat[i,j,:])
            # update srcimage
            self.computeSourceImageSignal(j)

        # update G
        self.compute_G_simon()

    ## M-noise step
    def updateNoiseVar(self):
        self.computeExpectError()
        self.sigma2_n = (1/self.T*self.expectError).flatten()

    ## M-NMF step
    def updateNMF(self, j, num_iter, b_updW=0):

        p_j = (self.s_hat[j]**2 + self.gamma[j])/(self.beta[j]/self.nu_v);
        WH = np.dot(self.W[j], self.H[j]) + self.eps;

        Error = np.zeros(num_iter);

        for i in range(num_iter):

            if b_updW:
                # Multiplicative update of W
                self.W[j] = self.W[j]*np.dot(WH**(-2)*p_j,np.transpose(self.H[j]))/(np.dot(1/WH,np.transpose(self.H[j])) + self.eps)

                # Normalization
                w = np.sum(self.W[j], axis=0) + self.eps;
                d = np.diag(1/w);
                self.W[j] = np.dot(self.W[j],d);
                d = np.diag(w);
                self.H[j] = np.dot(d,self.H[j]);

                WH = np.dot(self.W[j], self.H[j]) + self.eps;

            # Multiplicative update of H
            self.H[j] = self.H[j]*np.dot(np.transpose(self.W[j]),WH**(-2)*p_j)/(np.dot(np.transpose(self.W[j]),1/WH) + self.eps)

            WH = np.dot(self.W[j], self.H[j]) + self.eps;


            Error[i] = np.sum(np.sum(p_j/WH - np.log(p_j/WH + self.eps) + 1))

        return Error

    def updateLambda(self):
        num_iter_NMF = 20;
        for j in range(self.J):
            error = self.updateNMF(j, num_iter_NMF);
            self.lambda2[j] = np.dot(self.W[j],self.H[j])

    ### Other methods
    def computeExpectError(self):

        for i in range(self.I):
            norm_x_y = np.sum((self.x[:,i] - np.sum(self.s_im[i,:,:], axis=0))**2)
            print(self.s_im[i,:,:])
            norm_s_rho = np.sum(np.sum(self.sig_source**2, axis=1)*np.sum(self.rho[i,:,:], axis=1))
            print(self.sig_source)
            tmp = np.zeros(self.J);

            for j in range(self.J):
                normG = np.sum(self.G[j][:,:,i]**2, axis=1)
                tmp_val= (normG + np.sum(self.rho[i,j,:])).reshape(-1,1)
                tmp[j] = np.sum(np.sum(self.gamma[j] * tmp_val));
            sum_gamma_g_rho = np.sum(tmp)
            self.expectError[i] = norm_x_y + norm_s_rho + sum_gamma_g_rho
            print(self.expectError[i])

    def computeVFE(self):

        self.computeExpectError()

        likelihood_term = -self.I*self.T/2*np.log(2*np.pi) - self.T/2*np.sum(np.log(self.sigma2_n)) \
            - 1/2*np.sum(self.expectError/self.sigma2_n)

        s_term = np.sum(np.asarray(self.F)*np.asarray(self.N))/2 + np.sum(np.asarray(self.F)*np.asarray(self.N))/2*psi(self.nu_v)
        for j in range(self.J):
            s_term = s_term - 0.5*np.sum(np.log(self.beta[j].flatten()/(self.gamma[j].flatten() + self.eps) + self.eps) \
                + np.log(self.lambda2[j].flatten() + self.eps) \
                + self.nu_v/(self.beta[j].flatten() + self.eps)*(self.s_hat[j].flatten()**2 + self.gamma[j].flatten())/(self.lambda2[j].flatten() + self.eps))
        v_term = - np.sum(np.asarray(self.F)*np.asarray(self.N))*(gammaln(self.alpha_v/2) + self.alpha_v/2*np.log(2/self.alpha_v) \
            + psi(self.nu_v)*(self.nu_v - self.alpha_v/2) - self.nu_v - gammaln(self.nu_v))
        for j in range(self.J):
            v_term = v_term - self.alpha_v/2*np.sum(self.nu_v/self.beta[j].flatten() + np.log(self.beta[j].flatten()))

        a_term = self.I*self.J*self.La/2 - self.I*self.J/2*(np.sum(np.log(self.r2 + self.eps)) - np.sum(psi(self.nu_u))) \
            - 0.5*np.sum(np.sum(np.sum(np.log(self.d/(self.rho + self.eps) + self.eps) + self.nu_u.flatten()/self.r2*(self.a_hat**2 + self.rho)/self.d)))

        u_term = -self.I*self.J*np.sum(gammaln(self.alpha_u/2) + self.alpha_u/2*np.log(2/self.alpha_u) \
            + psi(self.nu_u)*(self.nu_u - self.alpha_u/2) - self.nu_u - gammaln(self.nu_u)) \
            - np.sum(self.alpha_u.flatten()/2*np.sum(np.sum(np.log(self.d)+ self.nu_u.flatten()/self.d, axis=0), axis=0))

        print(likelihood_term, s_term, v_term, a_term, u_term)
        self.varFreeEnergy = likelihood_term + s_term + v_term + a_term + u_term

    def compute_g_ijfn(self, i, j, f, n):
        philen = self.wlen[0]
        phi_fn = self.getMDCTAtom(f, n, philen, self.wlen[0]).reshape(1,-1)
        a_ij = self.a_hat[i,j,:]
        if type(i) == int:
            g_ijfn = convFFT(phi_fn, a_ij)
        else:
            g_ijfn = convFFT(np.dot(np.ones((len(i),1)),phi_fn),a_ij, axis=1)
        # n_0 = n*self.wlen[0]/2
        # g_ijfn = g_ijfn[n_0:n_0 + self.wlen[0] + self.La -1]
        print(np.mean(g_ijfn))

        return g_ijfn

    def compute_G(self):
        # for i in range(self.I):
        k = 0
        for j in range(self.J):
            for f in range(self.F[j]):
                self.G[:,k,:] = self.compute_g_ijfn(np.arange(self.I),j,f,0)
                k += 1

    def compute_G_simon(self):

        for j in range(self.J):
            win = np.sin((np.arange(self.wlen[j]) + 0.5)/self.wlen[j]*np.pi)
            a_sq = self.a_hat[:,j,:].T
            for f in range(self.F[j]):
                phi_f = win*np.sqrt(2/self.F[j])*np.cos((f + 0.5)*(np.arange(self.wlen[j]) + 0.5 + self.wlen[j]/4)*2*np.pi/self.wlen[j])
                self.G[j][f,:,:] = convFFT(np.dot(phi_f.T.reshape(-1,1), np.ones((1,self.I))), a_sq)

    def getGj(self,j):
        ## To change when wlen[j] != wlen[j-1]
        # Gj = self.G[:,j*self.F[j]:j*self.F[j] + self.F[j],:]
        Gj = self.G[j].transpose(2,0,1) # [ I x F x wlen + La - 1 ]

        return Gj

    def computeSourceSignal(self):

        for j in range(self.J):
            sig_j = imdct(self.s_hat[j], self.Ls)
            self.sig_source[j,:] = sig_j.flatten()

    def computeSourceImageSignal(self, j=None):
        if j != None:
            for i in range(self.I):
                self.s_im[i,j,:] = np.convolve(self.a_hat[i,j,:], self.sig_source[j,:]).flatten()
        else:
            for i in range(self.I):
                for j in range(self.J):
                    self.s_im[i,j,:] = np.convolve(self.a_hat[i,j,:], self.sig_source[j,:]).flatten()

    def computeWFilt(self, w):

        w_resh = np.zeros((self.Ls, self.J, self.I))
        for i in range(self.I):
            w_resh[:,:,i] = w
        w_resh = w_resh.reshape(self.Ls, self.I*self.J)
        a_resh = self.a_hat.T
        a_resh = a_resh.reshape(self.La, self.I*self.J)
        wi = convFFT(w_resh,a_resh)
        wi = wi.reshape(self.T,self.J,self.I)

        return wi

    def computePhiCorr(self):

        for j in range(self.J):

            if j != 0 and self.wlen[j] == self.wlen[j-1]:
                self.phiCorr[j] = self.phiCorr[j-1];

            else:
                phiCorr_j = np.zeros((self.F[j], self.La));
                for f in range(self.F[j]):
                    phi_jfn = self.getMDCTAtom(f, 0, self.Ls, self.wlen[j]);
                    phiCorr_j[f,:] = xcorr(phi_jfn, phi_jfn, self.La);
                self.phiCorr[j] = phiCorr_j;

    def getMDCTAtom(self, f, n, siglen, wlen):

        F = wlen/2
        win = np.sin(np.pi*np.arange(0.5,wlen+0.5)/wlen)
        hop = F
        frameInd = np.arange(n*hop, n*hop + wlen, dtype=int);
        if frameInd[-1] > siglen - 1:
            print('Out of signal')
        phi = np.zeros((siglen, 1)).flatten()
        phi[frameInd] = win*(np.sqrt(2/F)*np.cos((f + 0.5)*(np.arange(wlen) + 0.5 +wlen/4)*2*np.pi/wlen))

        return phi
