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
        W_num = np.dot((np.dot(W,H) + eps)**(-2)*P,np.transpose(H));
        W_det = np.dot((np.dot(W,H) + eps)**(-1),np.transpose(H)) + eps;
        W = W*W_num/W_det;

        # Normalization
        w = np.sum(W, axis=0);
        d = np.diag(1/w);
        W = np.dot(W,d);
        d = np.diag(w);
        H = np.dot(d,H);

        WH = np.dot(W,H) + eps;

        P = 1/((alpha_v*(V + eps)**(-1) + (WH + eps)**(-1))/(alpha_v + 1))

        H_num = np.dot(np.transpose(W),(np.dot(W,H) + eps)**(-2)*P);
        H_det = np.dot(np.transpose(W),(np.dot(W,H) + eps)**(-1)) + eps;
        H = H*H_num/H_det;

        WH = np.dot(W,H) + eps;

        P = 1/((alpha_v*(V + eps)**(-1) + (WH + eps)**(-1))/(alpha_v + 1))

        Error[i,0] = np.sum(np.sum(P/WH - np.log(P/WH + eps) - np.ones((F,N))))/S;

    return W, H, Error


def mdct(x, wlen):
    m = np.arange(wlen);
    win = np.sin((m*np.pi/(wlen - 1)));

    F = wlen/2; # frequency bins
    H = wlen/2; # hop size

    # padding x to wlen
    zeroPad_end = np.zeros((wlen - x.shape[0]%wlen + wlen/2,1));
    zeroPad_beg = np.zeros((wlen/2,1));
    xpad = np.concatenate((zeroPad_beg,x,zeroPad_end), axis=0);
    N = xpad.shape[0]/H - 1;
    X = np.zeros((F,N));
    for n in range(N):
        print(n, win.shape)
        x_win = xpad[n*H:n*H+wlen,0]*win;
        for f in range(F):
            X[f,n] = np.sum(x_win*np.cos(np.pi/(2*wlen)*(2*m + 1 + wlen/2)*(2*f + 1)));
    return X

def imdct(X, xlen):
    wlen = X.shape[0]*2;
    N = X.shape[1];
    n = np.arange(wlen);
    win = np.sin((n*np.pi/(wlen - 1)));
    F = wlen/2; # frequency bins
    x = np.zeros((xlen,1));
    m = np.arange(wlen/2);
    for n in range(N):
        for p in range(wlen):
            x[n*wlen/2 + p] += win[p]*4/wlen*np.sum(X[:,n]*np.cos(np.pi/(2*wlen)*(2*p + 1 + wlen/2)*(2*m + 1)))
    return x

def convFFT(x,y, axis=0):

    x_len = x.shape[axis];
    y_len = y.shape[axis];
    z_len = x_len + y_len - 1;
    z_pow2 = np.ceil(np.log2(z_len))
    fft_len = np.int(2**z_pow2)

    X = np.fft.fft(x, fft_len, axis=axis)
    Y = np.fft.fft(y, fft_len, axis=axis)
    z = np.imag(np.fft.ifft(X*Y, fft_len, axis=axis))
    
    if axis==0:
        z = z[:z_len,:]
    else:
        z = z[:,:z_len]

    return z

def xcorr(x, y, length):

    tmp = correlate(x,y,mode='same')
    n_0 = np.int(np.floor(len(tmp)/2))
    corr = tmp[n_0:n_0+length]

    return corr
