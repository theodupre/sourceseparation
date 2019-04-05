import numpy as np
import matplotlib.pyplot as plt
from utility_functions import *
from scipy.io import savemat, loadmat
from scipy.signal import correlate

# ## Test NMF et MDCT
# fe = 16000;
# t = np.linspace(0,2,2*fe);
# f1 = 800;
# f2 = 1500;
# f3 = 900;
# wlen = 2048;
# sig1 = np.sin(2*np.pi*f1*t);
# sig2 = np.sin(2*np.pi*f2*t);
# sig3 = np.sin(2*np.pi*f3*t);
# sig1 = np.concatenate((np.zeros(2*fe),sig1))
# sig2 = np.concatenate((np.zeros(1*fe),sig2,np.zeros(1*fe)))
# sig3 = np.concatenate((sig3,np.zeros(2*fe)))
#
#
#
# S1 = np.abs(mdct(sig1,wlen,zeropad=1))**2;
# S2 = np.abs(mdct(sig2,wlen,zeropad=1))**2;
# S3 = np.abs(mdct(sig3,wlen,zeropad=1))**2;
# S = S1 + S2 + S3;
#
# plt.figure()
# plt.imshow(S, aspect='auto', interpolation='None')
#
# [F,N] = S.shape
#
#
# K = 10;
# # W0 = 0.5 * (np.abs(np.random.randn(int(F),K)) + np.ones((int(F),K))) * \
# #         np.dot(np.mean(np.abs(S)**2, 1).reshape(-1,1),np.ones(K).reshape(1,-1));
# # H0 = 0.5 * (np.abs(np.random.randn(K,int(N))) + np.ones((K,int(N))));
# W0 = 0.5 * np.ones((F,K))
# H0 = 0.5 * np.ones((K,N))
# W1,H1,_ = nmf_mdct(S, 100, W0, H0, 200);
#
# WH = np.dot(W1,H1);
# savemat('/cal/exterieurs/atiam6576/Téléchargements/code-SSMM-MASS-2017/data/test1.mat', mdict={'WH': WH})
#
# plt.figure()
# plt.imshow(WH, aspect='auto', interpolation='None')
# plt.show()



# Test xcorr

file = "sounds/shorter_files/mix.wav";

epsi = loadmat('data/epsilon_ij.mat');
epsilon = epsi['epsilon_ij'];
print(epsilon.shape)
s = loadmat('data/s_j.mat')
s_j = s['s'];
print(s_j.shape)

La = 5761;

r_se = xcorr(epsilon,s_j,La, zeroPad=True)
print(len(r_se))
leng = len(r_se)
print(r_se)
plt.plot(r_se)
plt.show()


# La = 500
# sig1 =  np.concatenate((np.ones((200,1)),np.zeros((fe - 200,1))), axis=0)
#
# r_ss = xcorr(sig1,sig1,La)
# print(r_ss)
# plt.plot(r_ss)
# plt.show()

# Test imdct
# J = 3
# s_hat = loadmat('/cal/exterieurs/atiam6576/Téléchargements/code-SSMM-MASS-2017/data/s_hat.mat')
# s_hat = s_hat['s_hat'];
# s_h = []
# for j in range(J):
#     s_h.append(s_hat[j])
#     print(s_h[j].shape)
#
# s = imdct(s_h[0], 36864)
# plt.plot(s)
# plt.show()
#
# savemat('/cal/exterieurs/atiam6576/Téléchargements/code-SSMM-MASS-2017/data/s.mat', mdict={'s': s})

## test convFFT

# x1 = np.ones((30,1))
# x = np.concatenate((x1*1,x1*2,x1*3),axis=1)
#
# y1 = np.ones((20,1))
# y = np.concatenate((y1*1,y1*2,y1*3),axis=1)
#
# X = convFFT(x,y)
#
# plt.plot(X)
# plt.show()
