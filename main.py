import numpy as np
import scipy.io.wavfile as wav
from VEM import VEM
from utility_functions import *
import matplotlib.pyplot as plt
import time
from scipy.io import savemat,loadmat


file = "sounds/shorter_files/mix.wav";

fs, x = wav.read(file)
x = x.reshape((len(x),-1))
x = 0.99*x/np.max(x);

# General paramete
[T, I] = x.shape;
J = 3; # number of sources
sigma2_n = (np.sum(np.abs(x)**2,0)/T).flatten(); # noise variance
sigma = loadmat('data/sigma.mat')
sigma = sigma['sigma2_init']
sigma2_n = sigma.flatten()

# Mixing parameters
T60 = 0.360; # temps de reverb en s
La = np.int(T60*fs+1); # longueur d'un mixing filter
alpha_u = np.ones((La,1)); # Student's t shape mixing parameter
tau = T60*fs/(3*np.log(10)); # exponenetial decay rate of mixing filters
t = np.arange(La); # timeline in s of a mixing filter
r2 = 0.1**2*np.exp(-2*t/tau); # incomplete variance of mixing filter coefficients


# Source parameters
Ls = T - La + 1; # length of source signal
alpha_v = 100; # Student's t shape source parameter
wlen = [2048, 2048, 2048]; # window lengths for MDCT [drums, voice, bass]
K = 10; # NMF rank
num_iter = 200; # number of iteSettingsration for NMF optimization
W = []; # spectral templates
H = [np.ones((10,35))] * J; # activation matrices

## Zero padding do handle edges

hop = wlen[0]/2.
xpad = zeroPad(x, Ls, hop)


# for j in range(J):
#     filename = 'sounds/shorter_files/src' + str(j+1) + '.wav';
#     _, s = wav.read(filename);
#     s = s.reshape((len(s),-1))
#     s = zeroPad(s, Ls, hop)
#     S = mdct(s, wlen[j]);
#     # plt.imshow(np.abs(S)**2, aspect='auto', interpolation='None')
#     # plt.show()
#
#     [F,N] = S.shape;
#     W_init = 0.5*(np.abs(np.random.randn(F,K) + np.ones((F,K)))*np.dot(np.mean(np.abs(S)**2, axis=1).reshape((-1,1)),np.ones((1,K))))
#     H_init = 0.5*(np.abs(np.random.randn(K,N) + np.ones((K,N))));
#     H_init = 0.5*np.ones((K,N))
#     W_init = 0.5*np.ones((F,K))
#     Wj,Hj,_ = nmf_mdct(np.abs(S)**2, alpha_v, W_init, H_init, num_iter)
#
#     #Hj = H_init*2
#     WH = np.dot(Wj,Hj)
#     savemat('/cal/exterieurs/atiam6576/Téléchargements/code-SSMM-MASS-2017/data/WH' + str(j) + '.mat', mdict={'WH': WH})
#
#     # plt.imshow(Wj/np.max(Wj), aspect='auto', interpolation='none')
#     # plt.colorbar()
#     # plt.show()
#     W.append(Wj);
#     H.append(Hj); # H matrices will be estimated blindly


W_init = loadmat('data/W.mat');
W_init = W_init['W_init'];
for j in range(J):
    W.append(W_init[j][0])


niter = 200
varFreeEnergy = np.zeros(niter + 1)
print('Initialization of VEM algorithm...')
init = VEM(fs, xpad, alpha_u, alpha_v, sigma2_n, r2, W, H, wlen);
init.computeVFE()
print('VFE : ',init.varFreeEnergy/(init.T*init.I))
varFreeEnergy[0] = init.varFreeEnergy/(init.T*init.I)

for n in range(niter):
    print('VEM iteration ', n,' of ', niter)
    print('E-V step..')
    init.updateV()
    # # init.computeVFE()
    # # print('VFE : ',init.varFreeEnergy/(init.T*init.I))
    # # init.computeVFE()
    # # print('VFE : ',init.varFreeEnergy/(init.T*init.I))
    print('E-S step..')
    init.updateS()
    # # # init.computeVFE()
    # # ## print('VFE : ',init.varFreeEnergy/(init.T*init.I))
    # print('E-A step..')
    # init.updateA()
    print('E-U step..')
    init.updateU()
    # # init.computeVFE()
    # # print('VFE : ',init.varFreeEnergy/(init.T*init.I))
    print('M-NMF step..')
    init.updateLambda()
    # # init.computeVFE()
    # # print('VFE : ',init.varFreeEnergy/(init.T*init.I))
    print('M-noise step..')
    init.updateNoiseVar()
    init.computeVFE()
    varFreeEnergy[n + 1] = init.varFreeEnergy/(init.T*init.I)
    print('VFE : ',init.varFreeEnergy/(init.T*init.I))

plt.plot(varFreeEnergy)
plt.show()
Ls = 34240;
se = np.zeros((Ls,J))

for j in range(J):

    # se[:,j] = imdct(init.s_hat[j], Ls)
    plt.plot(init.sig_source[j,:])
    wav.write('sounds/shorter_files/res_source' + str(j+1) + '.wav', fs, init.sig_source[j,:])
    plt.show()


# # print(Error)
# plt.plot(x1)
# plt.plot(x_hat, 'g--')
# plt.show()
# x_test = xpad[:,0].reshape(-1,1)
# print(x_test)
# # plt.plot(x_test)
# X = mdct(x_test, 2048)
# # plt.plot(X[:,10])
# # print(X[:,10])
# x_exp = imdct(X,x_test.shape[0])
# # print(x_exp.shape)
# #plt.imshow(X**2, aspect='auto', interpolation='none')
# # x = x_test.flatten() - x_exp
# # plt.plot(x_test)
# # plt.plot(x_exp)
# # plt.show()
