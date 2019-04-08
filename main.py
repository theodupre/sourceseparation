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
b_noiseAnnealing = True # noise update strategy
b_updW = 1 # update W (1) or not (0)

# Mixing parameters
T60 = 0.360; # temps de reverb en s
La = np.int(T60*fs+1); # longueur d'un mixing filter
alpha_u = np.ones((La,1)); # Student's t shape mixing parameter
tau = T60*fs/(3*np.log(10)); # exponenetial decay rate of mixing filters
t = np.arange(La); # timeline in s of a mixing filter
r2 = 0.1**2*np.exp(-2*t/tau); # incomplete variance of mixing filter coefficients

# Source parameters
Ls = T - La + 1 # length of source signal
alpha_v = 100 # Student's t shape source parameter
wlen = [512, 2048, 2048] # window lengths for MDCT [drums, voice, bass]
K = 10 # NMF rank
num_iter = 20 # number of iteSettingsration for NMF optimization
W = [] # spectral templates
H = [] # activation matrices

## Zero padding do handle edges
hop = wlen[0]/2.
xpad = zeroPad(x, Ls, hop)

# Noise annealing scheduling
if b_noiseAnnealing:
    sigma2_beg = 0.01
    sigma2_end = 0.000001
    sigma2_init = sigma2_beg*np.ones(I)
else:
    sigma2_int = (np.sum(np.abs(x)**2,0)/T).flatten(); # noise variance



for j in range(J):
    filename = 'sounds/shorter_files/src' + str(j+1) + '.wav';
    _, s = wav.read(filename);
    s = s.reshape((len(s),-1))
    s = zeroPad(s, Ls, hop)
    S = mdct(s, wlen[j]);

    [F,N] = S.shape;
    H.append(np.ones((K,N)));
    W_init = 0.5*(np.abs(np.random.randn(F,K) + np.ones((F,K)))*np.dot(np.mean(np.abs(S)**2, axis=1).reshape((-1,1)),np.ones((1,K))))
    H_init = 0.5*(np.abs(np.random.randn(K,N) + np.ones((K,N))));

    Wj,Hj,_ = nmf_mdct(np.abs(S)**2, alpha_v, W_init, H_init, num_iter)

    WH = np.dot(Wj,Hj)

    W.append(Wj);


niter = 25
varFreeEnergy = np.zeros(niter + 1)
print('Initialization of VEM algorithm...')
ss_VEM = VEM(fs, xpad, alpha_u, alpha_v, sigma2_init, r2, W, H, wlen);
ss_VEM.computeVFE()
print('VFE : ',ss_VEM.varFreeEnergy/(ss_VEM.T*ss_VEM.I))
varFreeEnergy[0] = ss_VEM.varFreeEnergy/(ss_VEM.T*ss_VEM.I)

for n in range(niter):
    print('VEM iteration ', n,' of ', niter)

    print('E-V step..')
    ss_VEM.updateV()

    print('E-S step..')
    ss_VEM.updateS()

    print('E-U step..')
    ss_VEM.updateU()

    print('E-A step..')
    ss_VEM.updateA()

    print('M-NMF step..')
    ss_VEM.updateLambda(b_updW)

    print('M-noise step..')
    if b_noiseAnnealing:
        ss_VEM.updateNoiseVarAnneal(n,niter, sigma2_beg, sigma2_end)
    else:
        ss_VEM.updateNoiseVar()

    ss_VEM.computeVFE()
    varFreeEnergy[n + 1] = ss_VEM.varFreeEnergy/(ss_VEM.T*ss_VEM.I)
    print('VFE : ',ss_VEM.varFreeEnergy/(ss_VEM.T*ss_VEM.I))

    if np.mod(n,10) == 0:
        for j in range(J):
            wav.write('sounds/shorter_files/results/res_source_' + str(n + 1) + '_' + str(j+1) + '.wav', fs, ss_VEM.sig_source[j,:])
plt.plot(varFreeEnergy)
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
