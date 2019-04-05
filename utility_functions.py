import numpy as np
import time
from scipy.signal import correlate

def nmf_mdct(V, alpha_v, W_init, H_init, num_iter):

    [F, N] = V.shape;

    H = H_init;
    W = W_init;

    Error = np.zeros((num_iter,1));
    eps = np.finfo(float).eps;

    WH = np.dot(W,H) + eps;

    P = 1/((alpha_v*(V + eps)**(-1) + (WH + eps)**(-1))/(alpha_v + 1))
    #P = V + eps;
    S = np.sum(P)

    for i in range(num_iter):

        # Multiplicative update for H
        W_num = np.dot((WH)**(-2)*P,np.transpose(H)) + eps;
        W_det = np.dot((WH)**(-1),np.transpose(H)) + eps;
        W = W*W_num/W_det;

        # Normalization
        w = np.sum(W, axis=0);
        d = np.diag(1/w);
        W = np.dot(W,d);
        d = np.diag(w);
        H = np.dot(d,H);

        WH = np.dot(W,H) + eps;

        P = 1/((alpha_v*(V + eps)**(-1) + (WH + eps)**(-1))/(alpha_v + 1))

        H_num = np.dot(np.transpose(W),(WH)**(-2)*P) + eps;
        H_det = np.dot(np.transpose(W),(WH)**(-1)) + eps;
        H = H*H_num/H_det;

        WH = np.dot(W,H) + eps;

        P = 1/((alpha_v*(V + eps)**(-1) + (WH + eps)**(-1))/(alpha_v + 1))

        Error[i,0] = np.sum(np.sum(P/WH - np.log(P/WH + eps) - np.ones((F,N))))/S;

    return W, H, Error


def mdct(x, wlen, zeropad=0):
    if zeropad != 0:

        x = zeroPad(x, len(x), 1024)

    m = np.arange(wlen);
    win = np.sin(((m + 0.5)*np.pi/wlen));
    F = int(wlen/2); # frequency bins
    H = F; # hop size
    N = int(x.shape[0]/H - 1);
    frameInd = np.zeros((wlen,N), dtype=int)
    for n in range(N):
        frameInd[:,n] = n*H + np.arange(wlen)
    X = np.zeros((F,N));

    T = np.zeros((F,wlen))
    for f in range(F):

        T[f,:] = win*np.sqrt(4/wlen)*np.cos(2*np.pi/wlen*(m + 0.5 + wlen/4)*(f + 0.5))

    X = np.dot(T,np.squeeze(x[frameInd]))

    return X


def imdct(X, xlen):
    wlen = X.shape[0]*2;
    N = X.shape[1];
    m = np.arange(wlen);
    win = np.sin(((m + 0.5)*np.pi/wlen));
    F = int(wlen/2); # frequency bins
    H = F
    x = np.zeros(xlen);

    T = np.zeros((F,wlen))
    for f in range(F):
        T[f,:] = win*np.sqrt(4/wlen)*np.cos((f + 0.5)*(m + 0.5 + wlen/4)*2*np.pi/wlen)

    frames = np.dot(T.T,X)
    frameInd = np.zeros((F,N))
    for n in range(N):
        frameInd = n*H + np.arange(wlen, dtype=int)
        x[frameInd] = x[frameInd] + frames[:,n]

    return x

def convFFT(x,y, axis=0):

    x_len = x.shape[axis];
    y_len = y.shape[axis];
    z_len = x_len + y_len - 1;
    z_pow2 = np.ceil(np.log2(z_len))
    fft_len = np.int(2**z_pow2)


    X = np.fft.rfft(x, fft_len, axis=axis)
    Y = np.fft.rfft(y, fft_len, axis=axis)
    z = np.fft.irfft(X*Y, fft_len, axis=axis)
    if axis==0:
        z = z[:z_len,:]
    else:
        z = z[:,:z_len]

    return z

def xcorr(x, y, length, zeroPad=False):

    if zeroPad:
        x_len = len(x)
        y_len = len(y)
        zeros = int(np.abs(np.floor((x_len - y_len))))
        if x_len > y_len:
            # y = np.concatenate((np.zeros(zeros), y.flatten(), np.zeros(zeros)))
            y = np.concatenate((y.flatten(),np.zeros(zeros)))
            x = x.flatten()
        else:
            # x = np.concatenate((np.zeros(zeros), x.flatten(), np.zeros(zeros)))
            x = np.concatenate((np.zeros(zeros), x.flatten()))
            y = y.flatten()

    tmp = correlate(x,y,mode='full')
    n_0 = np.int(np.floor(len(tmp)/2))
    corr = tmp[n_0:n_0+length]

    return corr

def zeroPad(x, Ls, hop):
    x = x.reshape(x.shape[0],-1)
    I = x.shape[1]
    zeroPad_end = np.zeros(((np.ceil(Ls/hop)*hop - Ls + hop).astype(int),I));
    zeroPad_beg = np.zeros((int(hop),I));
    xpad= np.concatenate((zeroPad_beg,x,zeroPad_end), axis=0);

    return xpad
