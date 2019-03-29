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



class VEM:

    def __init__(self, fs, x, alpha_u, alpha_v, sigma2_n, r2, W, H, wlen):

        self.eps = np.finfo(float).eps

        self.fs = fs; # sample_rate
        self.x = x;
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
        self.F = [];
        self.N = [];

        self.sig_source = np.empty((self.J, self.Ls));
        self.phiCorr = [];
        self.s_im = np.empty((self.I,self.J,self.T));
        self.expectError = np.empty(self.I)

        for j in range(self.J):
            self.F.append(int(self.wlen[j]/2));
            self.N.append(int(np.round(self.Ls/self.F[j]) - 1));
            self.phiCorr.append(np.empty([1,1]));

        self.lambda2 = [];
        for j in range(self.J):
            self.lambda2.append(np.dot(self.W[j],self.H[j]));

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
        for j in range(self.J):
            gamma_j = invgamma.rvs(self.nu_v, scale=self.beta[j])*self.lambda2[j];
            self.gamma.append(gamma_j);
            s_hat_j = np.random.randn(self.F[j],self.N[j])*gamma_j;
            self.s_hat.append(s_hat_j);

        # U
        self.nu_u = (self.alpha_u + 1)/2;
        d_temp = np.repeat((self.alpha_u/2).reshape((1,1,-1)),self.I,axis=0)
        self.d = np.repeat(d_temp, self.J, axis=1);

        # A
        self.a_hat = np.zeros((self.I,self.J,self.La));
        self.rho = self.a_hat;
        for i in range(self.I):
            for j in range(self.J):
                rho_ij = invgamma.rvs(self.nu_v, scale=self.d[i,j,:])*self.r2;
                self.rho[i,j] = rho_ij;
                self.a_hat[i,j] = np.random.randn(self.La)*rho_ij;

        self.G = np.zeros((self.I, self.J*self.F[0],self.wlen[0] + self.La - 1))


        self.compute_G();
        self.computePhiCorr();
        self.computeSourceSignal();
        self.computeSourceImageSignal();

    ## E-V step
    def updateV(self):
        for j in range(self.J):
            self.beta[j] = self.alpha_v/2 + (self.s_hat[j]**2 + self.gamma[j])/(2*self.lambda2[j] + self.eps);

    ## E-U step
    def updateU(self):
        for i in range(self.I):
            for j in range(self.J):
                self.d[i,j] =(self.alpha_u/2).ravel() + (self.a_hat[i,j]**2 + self.rho[i,j])/(2*self.r2 + self.eps)

    ## E-S step
    def updateS(self):
        ##  Update gamma

        inv_v_post = [None] * self.J
        for j in range(self.J):
            inv_v_post[j] = self.nu_v/self.beta[j]
            norm2_g = np.sum(self.getGj(j)**2, axis=2)
            sum_rho = np.sum(self.rho[:,j,:], axis=1). reshape(-1,1)
            sum_i = np.sum(1/(self.sigma2_n + self.eps)*(sum_rho + norm2_g), axis=0).reshape(-1,1)
            self.gamma[j] = inv_v_post[j]/(self.lambda2[j] + self.eps) + sum_i

        ## Update s with pcg algorithm
        # Solving [LAMBDA]s = GX

        # compute gradient
        epsilon = self.x - np.sum(self.s_im, axis=1).T
        inv_sigma2_sum_rho = np.sum(1/self.sigma2_n * np.sum(self.rho, axis=2),axis=0)
        frameInd = [None] * self.J
        sum_GX = [None] * self.J
        grad = [None] * self.J
        for j in range(self.J):
            hop = self.wlen[j]/2
            frameInd[j] = np.zeros((self.wlen[j] + self.La - 1, self.N[j]), dtype=int)
            for n in range(self.N[j]):
                frameInd[j][:,n] = np.arange(n*hop, n*hop + self.wlen[j] + self.La - 1)

            epsilonMat = np.zeros((self.I, self.wlen[j] + self.La - 1, self.N[j]))

            for i in range(self.I):
                epsilon_i = epsilon[:,i]/self.sigma2_n[i]
                epsilonMat[i,:,:] = epsilon_i[frameInd[j]]

            grad[j] = self.s_hat[j]*(inv_v_post[j]/(self.lambda2[j] + self.eps) + inv_sigma2_sum_rho[j]) - np.sum(np.matmul(self.getGj(j), epsilonMat), axis=0)

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
                    w_filtMat[i,:,:,] = w_filt_i[frameInd[j]]

                kappa[j] = w[j]*(inv_v_post[j]/(self.lambda2[j] + self.eps) + inv_sigma2_sum_rho[j]) + np.sum(np.matmul(self.getGj(j),w_filtMat), axis=0)

            # comput mu
            mu = np.matmul(np.asarray(w).flatten().T,np.asarray(grad).flatten())/np.matmul(np.asarray(w).flatten().T,np.asarray(kappa).flatten())

            # update paramaters
            for j in range(self.J):
                self.s_hat[j] = self.s_hat[j] - mu*w[j]

            # update sources
            self.computeSourceSignal()
            self.computeSourceImageSignal()

            if iter == niter:
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

            # compute
            for j in range(self.J):
                w[j] = gradP[j] + alpha_pcg*w[j]
    ## E-A step
    def updateA(self):

        # update source_signal
        self.computeSourceSignal();

        # update rho
        norm2_s = [];
        for j in range(self.J):
            sum_rho = np.sum(np.sum(self.gamma[j]));
            norm2_s = np.sum(self.sig_source[j,:]**2);
            sigma_term = (sum_rho + norm2_s)*1/self.sigma2_n
            nu_term = np.transpose(self.nu_u)/(self.d[:,j,:]*self.r2)
            self.rho[:,j,:] = nu_term + sigma_term

        ## compute auxiliary functions
        # autoCorr de S_hat
        r_ss = []
        for j in range(self.J):
            r_ss_j = xcorr(self.sig_source[j,:], self.sig_source[j,:], self.La)
            r_ss.append(r_ss_j)

        # sum_gamma_r_phiphi et toeplitz matrix
        gam_r = np.zeros((self.La, 1))
        toeplitz_mat = np.empty((self.I,self.J,self.La,self.La))
        for j in range(self.J):
            gamma_jfn = self.gamma[j];
            for t in range(self.La):
                phiCorr_t = self.phiCorr[j][:,t].reshape(-1,1);
                phiCorr_t = np.tile(phiCorr_t, [1,self.N[j]]);
                gam_r[t,0] = np.dot(phiCorr_t.reshape(1,-1), gamma_jfn.reshape(-1,1))
            toep_diags = r_ss[j] + gam_r[:,0];
            toeplitz_mat[0,j,:,:] = toeplitz(toep_diags);

        for i in range(self.I):
            toeplitz_mat[i,:,:,:] = 1/self.sigma2_n[i]*toeplitz_mat[0,:,:,:]


        lambda_a = np.empty((self.I,self.J,self.La,self.La))
        epsilon_ij = np.empty((self.I, self.J, self.T))
        r_se = np.empty((self.I, self.J, self.La))
        r_se_sigma = np.empty(r_se.shape)
        for j in range(self.J):
            for i in range(self.I):

                # Lambda_a
                tmp = (self.d[i,j,:]*self.r2).reshape(-1,1)
                diag = np.diag((self.alpha_u/tmp).flatten())
                lambda_a[i,j,:,:] = diag + toeplitz_mat[i,j,:,:]

                # epsilon_ij et r_se/sigma2_n
                sum_y_ij = np.sum(self.s_im[i,:,:], axis=0) - self.s_im[i,j,:]
                epsilon_ij[i,j,:] = self.x[:,i] - sum_y_ij
                r_se[i,j,:] = 1/self.sigma2_n[i]*xcorr(self.sig_source[j,:].reshape(-1,1), epsilon_ij[i,j,:].reshape(-1,1), self.La).flatten()

                precondA_hat = np.diag(1/np.diag(lambda_a[i,j,:,:]))

                ## Compute preconditionned conjugate gradient descent
                self.a_hat[i,j,:],conv = cg(lambda_a[i,j,:,:], r_se[i,j,:].reshape(-1,1), M=precondA_hat, maxiter=10)


            # update srcimage
            self.computeSourceImageSignal();

        # update PhiCorr
        self.computePhiCorr();

    ## M-noise step
    def updateNoiseVar(self):
        self.sigma2_n = (1/self.T*self.expectError).reshape(-1,1)

    ## M-NMF step
    def updateNMF(self, j, num_iter):

        p_j = (self.s_hat[j]**2 + self.gamma[j])/(self.beta[j]/self.alpha_v);
        WH = np.dot(self.W[j], self.H[j]) + self.eps;

        Error = np.zeros(num_iter);

        for i in range(num_iter):

            # Multiplicative update of W
            self.W[j] = self.W[j]*np.dot(WH**(-2)*p_j,np.transpose(self.H[j]))/(np.dot(1/WH,np.transpose(self.H[j])) + self.eps)

            # Normalization
            w = np.sum(self.W[j], axis=0);
            d = np.diag(1/w);
            self.W[j] = np.dot(self.W[j],d);
            d = np.diag(w);
            self.H[j] = np.dot(d,self.H[j]);

            WH = np.dot(self.W[j], self.H[j]) + self.eps;

            # Multiplicative update of H
            self.H[j] = self.H[j]*np.dot(np.transpose(self.W[j]),WH**(-2)*p_j)/(np.dot(np.transpose(self.W[j]),1/WH) + self.eps)

            WH = np.dot(self.W[j], self.H[j]) + self.eps;

            Error[i] = np.sum(np.sum(p_j/WH - np.log(p_j/WH) + 1))

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
            norm_s_rho = np.sum(np.sum(self.sig_source**2)*np.sum(self.rho[i,:,:], axis=1))

            tmp = np.zeros(self.J);
            for j in range(self.J):
                tmp_val= (np.sum(self.getGj(j)[i,:,:]**2, axis=1) + np.sum(self.rho[i,j,:])).reshape(-1,1)
                tmp[j] = np.sum(np.sum(self.gamma[j] * tmp_val));
            sum_gamma_g_rho = np.sum(tmp)

            self.expectError[i] = norm_x_y + norm_s_rho + sum_gamma_g_rho

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

        return g_ijfn

    def compute_G(self):
        # for i in range(self.I):
        k = 0
        for j in range(self.J):
                # t = [i+0.5 for i in range(self.wlen[j])];
                # t = np.array((t));
                # win = np.sin((t*np.pi/self.wlen[j]));
                # a_temp = self.a_hat[:,j,:]
            for f in range(self.F[j]):
                # phi_nf = win*(np.sqrt(4/self.F[j])*np.cos((f + 0.5) * (t + 0.5 + self.wlen[j]/4)))
                    # self.G[j][:,f,:] = convFFT(np.dot(np.ones((self.I,1)), phi_f.reshape(1,-1)),a_temp, axis=1)
                self.G[:,k,:] = self.compute_g_ijfn(np.arange(self.I),j,f,0)
                k += 1

    def getGj(self,j):
        ## To change when wlen[j] != wlen[j-1]
        Gj = self.G[:,j*self.F[j]:j*self.F[j] + self.F[j],:]
        return Gj

    def computeSourceSignal(self):

        for j in range(self.J):
            sig_j = imdct(self.s_hat[j], self.Ls);
            self.sig_source[j,:] = sig_j.flatten()

    def computeSourceImageSignal(self):

        for i in range(self.I):
            for j in range(self.J):
                self.s_im[i,j,:] = convFFT(self.a_hat[i,j,:].reshape(-1,1), self.sig_source[j,:].reshape(-1,1)).flatten()

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
