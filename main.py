import numpy as np
import scipy.io.wavfile as wav
from VEM import VEM
from utility_functions import mdct,imdct,nmf_mdct,convFFT
# import matplotlib.pyplot as plt
import time


file = "sounds/shorter_files/mix.wav";

fs, x = wav.read(file)
x = x.reshape((len(x),-1))
x = 0.99*x/np.max(x);

# General paramete
[T, I] = x.shape;
J = 3; # number of sources
sigma2_n = 0.01*np.ones((I,1)); # noise variance

# Mixing parameters
T60 = 0.360; # temps de reverb en s
La = np.int(T60*fs+1); # longueur d'un mixing filter
alpha_u = np.ones((La,1)); # Student's t shape mixing parameter
tau = T60*fs/(3*np.log(10)); # exponenetial decay rate of mixing filters
t = np.linspace(0,La/fs, La); # timeline in s of a mixing filter
r2 = (0.1*np.exp(-t/tau))**2; # incomplete variance of mixing filter coefficients

# Source parameters
Ls = T - La + 1; # length of source signal
alpha_v = 100; # Student's t shape source parameter
wlen = [2048, 2048, 2048]; # window lengths for MDCT [drums, voice, bass]
K = 10; # NMF rank
num_iter = 100; # number of iteSettingsration for NMF optimization
W = []; # spectral templates
H = []; # activation matrices
#
## Zero padding do handle edges

hop = wlen[0]/2.
zeroPad_end = np.zeros(((np.ceil(Ls/hop)*hop - Ls + hop).astype(int),I));
zeroPad_beg = np.zeros((int(hop),I));
xpad= np.concatenate((zeroPad_beg,x,zeroPad_end), axis=0);


for j in range(J):

    W.append(np.load('data/shorter_files/W_init_' + str(j+1) + '.npy'))
    H.append(np.load('data/shorter_files/H_init_' + str(j+1) + '.npy'))
# start_time = time.time()
# initialization Wj dictionaries using NMF on already isolated sources
# for j in range(J):
#     filename = 'sounds/shorter_files/src' + str(j+1) + '.wav';
#     _, s = wav.read(filename);
#     s = s.reshape((len(s),-1))
#     S = mdct(s, wlen[j]);
#
#     [F,N] = S.shape;
#     W_init = 0.5 * (np.abs(np.random.randn(F,K) + np.ones((F,K)))*np.dot(np.mean(np.abs(S)**2, axis=1).reshape((-1,1)),np.ones((1,K))))
#     H_init = 0.5*(np.abs(np.random.randn(K,N) + np.ones((K,N))));
#     print(W_init.shape, H_init.shape)
#     Wj,_,_ = nmf_mdct(np.abs(S)**2, alpha_v, W_init, H_init, num_iter)
#
#     np.save('data/shorter_files/W_init_' + str(j+1) + '.npy', Wj)
#     np.save('data/shorter_files/H_init_' + str(j+1) + '.npy', H_init)
#
#     W.append(Wj);
#     H.append(H_init); # H matrices will be estimated blindly

# print()

# x = np.ones((wlen[0], 3))
# y = np.ones((La, 3))
#
# z = convFFT(x,y)
# print(z.shape, La + wlen[0] - 1)

# W0,H0,Error = nmf_mdct()
# plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(W/np.max(W), aspect='auto', interpolation='none')
# plt.colorbar()
# plt.subplot(1,2,2)
# plt.imshow(H/np.max(H), aspect='auto', interpolation='none')

# plt.colorbar()
# plt.figure()
# plt.subplot(2,1,1)
# plt.imshow(np.abs(X)**2/np.max(np.abs(X)**2), aspect='auto', interpolation='none')
# #plt.imshow(np.abs(X)/np.max(X))
# plt.subplot(2,1,2)
# plt.imshow(np.dot(W,H)/np.max(np.dot(W,H)), aspect='auto', interpolation='none')
# plt.figure();
# plt.plot(Error)

#plt.matshow(abs(X), origin='upper',cmap='gray');
#plt.show();
# #x_hat = imdct(np.abs(X));
init = VEM(fs, xpad, alpha_u, alpha_v, sigma2_n, r2, W, H, wlen);
#init.updateV()
#init.updateU()
init.updateS()# t = np.linspace(0,T/fs,T);
#init.updateA()
#init.updateLambda()
Error = init.computeExpectError();
# print(Error)
# plt.plot(x1)
# plt.plot(x_hat, 'g--')
# plt.show()
# x_test = x[:,0].reshape(-1,1)
# print(x_test.shape)
# x_exp = imdct(mdct(x_test,2048),x_test.shape[0])
# print(x_exp.shape)
