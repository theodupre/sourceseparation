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

        self.T = self.x.shape[0];
        self.I = self.x.shape[1];
        self.J = len(self.W);
        self.La = self.alpha_u.shape[0];
        self.Ls = self.T - self.La + 1;
        self.wlen = wlen;
        self.F = [];
        self.N = [];

        self.sig_source = np.empty((self.J, self.Ls));
        self.phiCorr = [];
        self.s_im = np.empty((self.I,self.J,self.T));

        for j in range(self.J):
            self.F.append(self.wlen[j]/2);
            self.N.append(np.round(self.Ls/self.F[j]) - 1);
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
        print(self.d.shape)

        # A
        self.a_hat = np.zeros((self.I,self.J,self.La));
        self.rho = self.a_hat;
        for i in range(self.I):
            for j in range(self.J):
                rho_ij = invgamma.rvs(self.nu_v, scale=self.d[i,j,:])*self.r2;
                self.rho[i,j] = rho_ij;
                self.a_hat[i,j] = np.random.randn(self.La)*rho_ij;

        self.G = []
        for j in range(self.J):
            G_j = np.zeros((self.I,self.F[j],self.La + self.wlen[j] - 1))
            self.G.append(G_j)

        self.compute_G();
        self.computePhiCorr();
        self.computeSourceSignal();
        self.computeSourceImageSignal();
        print('phicorr', self.phiCorr[0].shape)
        print('G', self.G[0].shape)

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
        for j in range(self.J):
            norm2_g = np.sum(self.G[j]**2, axis=2)
            sum_rho = np.sum(self.rho[:,j,:], axis=1). reshape(-1,1)
            sum_i = np.sum(1/(self.sigma2_n + self.eps)*(sum_rho + norm2_g), axis=0).reshape(-1,1)
            self.gamma[j] = self.nu_v/(self.beta[j]*self.lambda2[j] + self.eps) + sum_i


        ## Update s with pcg algorithm
        # Solving [LAMBDA]s = GX

        # GX [J][FN x 1]
        frameInd = []
        sum_GX = []
        for j in range(self.J):
            hop = self.wlen[j]/2
            frameInd_j = np.zeros((self.wlen[j] + self.La - 1, self.N[j]))
            print('n_max', np.max(self.N[j]))
            for n in range(self.N[j]):
                frameInd_j[:,n] = np.arange(n*hop, n*hop + self.wlen[j] + self.La - 1)
            frameInd.append(frameInd_j);
            GX = np.zeros((self.wlen[j] + self.La - 1, self.N[j], self.I))
            for i in range(self.I):
                X_i = self.x[:,i]/(self.sigma2_n[i] + self.eps);
                print(X_i.shape, frameInd[j][:,-1:])
                GX[:,:,i] = np.dot(self.G[j][i,:,:], X_i[frameInd[j].astype(int)])
            sum_GX.append(np.sum(GX, axis=2).flatten());

        # LAMBDA
        inv_sigma_rho = []
        for j in range(self.J):
            inv_sigma_rho.append(np.sum(1/self.sigma2_n + np.sum(self.rho[:,j,:], axis=1)))
            nu_beta_lambda = self.nu_v/(self.beta[j]*self.lambda2[j] + self.eps)

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
        print('toeplitz_mat', toeplitz_mat.shape)


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
            print (error)

    ### Other methods
    def computeExpectError(self):

        expectError = np.empty(self.I)
        for i in range(self.I):
            norm_x_y = np.sum((self.x[:,i] - np.sum(self.s_im[i,:,:], axis=0))**2)
            norm_s_rho = np.sum(np.sum(self.sig_source**2)*np.sum(self.rho[i,:,:], axis=1))

            tmp = np.zeros(self.J);
            for j in range(self.J):
                tmp_val= (np.sum(self.G[j][i,:,:]**2, axis=1) + np.sum(self.rho[i,j,:])).reshape(-1,1)
                tmp[j] = np.sum(np.sum(self.gamma[j] * tmp_val));
            sum_gamma_g_rho = np.sum(tmp)

            expectError[i] = norm_x_y + norm_s_rho + sum_gamma_g_rho

        return expectError


    def compute_G(self):

        for j in range(self.J):
            t = [i+0.5 for i in range(self.wlen[j])];
            t = np.array((t));
            win = np.sin((t*np.pi/self.wlen[j]));
            a_temp = self.a_hat[:,j,:]
            for f in range(self.F[j]):
                phi_f = win*(np.sqrt(4/self.F[j])*np.cos((f + 0.5) * (t + 0.5 + self.wlen[j]/4)))
                self.G[j][:,f,:] = convFFT(np.dot(np.ones((self.I,1)), phi_f.reshape(1,-1)),a_temp, axis=1)

    def computeSourceSignal(self):

        for j in range(self.J):
            sig_j = imdct(self.s_hat[j], self.Ls);
            self.sig_source[j,:] = sig_j.flatten()

    def computeSourceImageSignal(self):

        for i in range(self.I):
            for j in range(self.J):
                self.s_im[i,j,:] = convFFT(self.a_hat[i,j,:].reshape(-1,1), self.sig_source[j,:].reshape(-1,1)).flatten()

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
        hop = wlen/2
        frameInd = np.arange(n*hop, n*hop + wlen);
        if frameInd[-1] > siglen - 1:
            print('Out of signal')
        phi = np.zeros((siglen, 1)).flatten()
        phi[frameInd] = win*(np.sqrt(2/F)*np.cos((f + 0.5)*(np.arange(wlen) + 0.5 +wlen/4)*2*np.pi/wlen))

        return phi
