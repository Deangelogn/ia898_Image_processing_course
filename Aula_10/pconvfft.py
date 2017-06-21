
# coding: utf-8

# 
# # Implementação da convolução periódica eficiente
# 
# Esta implementação deve ser melhor que a implementação atual da ia898 e mais eficiente,
# implementando a convolução no domínio da frequência caso necessário.

# In[1]:

import numpy as np
import matplotlib.image as mpimg
import sys, os
ia898path = os.path.abspath('/home/lotufo')
if ia898path not in sys.path:
    sys.path.append(ia898path)
import ia898.src as ia


# In[2]:

def pconvfft(f,h, th=15):
    '''
    Periodical convolution.
    This is an efficient implementation of the periodical convolution.
    This implementation should be comutative, i.e., pconvfft(f,h)==pconvfft(h,f).
    This implementation should be fast. If the number of pixels used in the 
    convolution is larger than 15, it uses the convolution theorem to implement
    the convolution.
    Parameters:
    -----------
        f: input image (can be complex, up to 2 dimensions)
        h: input kernel (can be complex, up to 2 dimensions)
    Outputs:
        image of the result of periodical convolution
    '''
    if h.size < f.size:
        nzMin = len(np.nonzero(h)[0])
    else:
        nzMin = len(np.nonzero(f)[0])
    
    # convolution in frquency domain
    
    if nzMin>th:
        if f.ndim == 1:
            if h.size <= f.size:
                hc = np.zeros_like(f)
                hc[:h.size] = h
                F = np.fft.fft(f)
                Hc = np.fft.fft(hc)
                return np.fft.ifft(F*Hc).real
            else:
                hc = np.zeros_like(h)
                hc[:f.size] = f
                F = np.fft.fft(h)
                Hc = np.fft.fft(hc)
                return np.fft.ifft(F*Hc).real
                
        elif f.ndim == 2:
            if h.size <= f.size:
                hc = np.zeros(f.shape)
                hc[:h.shape[0],:h.shape[1]] = h 
                F = np.fft.fft2(f)
                Hc = np.fft.fft2(hc)
                return np.fft.ifft2(F*Hc).real
            else:
                hc = np.zeros(h.shape)
                hc[:f.shape[0],:f.shape[1]] = f 
                F = np.fft.fft2(h)
                Hc = np.fft.fft2(hc)
                return np.fft.ifft2(F*Hc).real
            
    # old iimplementation  
    h_ind=np.nonzero(h)
    f_ind=np.nonzero(f)
    if len(h_ind[0])>len(f_ind[0]):
        h,    f    = f,    h
        h_ind,f_ind= f_ind,h_ind

    gs = np.maximum(np.array(f.shape),np.array(h.shape))
    if (f.dtype == 'complex') or (h.dtype == 'complex'):
        g = np.zeros(gs,dtype='complex')
    else:
        g = np.zeros(gs)

    f1 = g.copy()
    f1[f_ind]=f[f_ind]      

    if f.ndim == 1:
        (W,) = gs
        col = np.arange(W)
        for cc in h_ind[0]:
            g[:] += f1[(col-cc)%W] * h[cc]
            
    elif f.ndim == 2:
        H,W = gs
        row,col = np.indices(gs)
        for rr,cc in np.transpose(h_ind):
            g[:] += f1[(row-rr)%H, (col-cc)%W] * h[rr,cc]
            
    else:
        Z,H,W = gs
        d,row,col = np.indices(gs)
        for dd,rr,cc in np.transpose(h_ind):
            g[:] += f1[(d-dd)%Z, (row-rr)%H, (col-cc)%W] * h[dd,rr,cc]
    return g


# In[3]:

testing = (__name__ == '__main__')

if testing:
    import sys,os
    get_ipython().system(" jupyter nbconvert --to 'python' pconvfft")

    path = os.path.abspath('/etc/jupyterhub/ia898_1s2017/d191122/Aula_10/')
    if path not in sys.path:
        sys.path.append(path)
    import pconvfft as conv


# In[ ]:

if testing:
    # Teste 1D conv
    f = np.array([0,0,0,1,0,0,0,0,1])
    f = np.array([f,f,f,f,f,f])
    f = f.ravel()
    h = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])

    print('f:\n',f)
    print('h:\n',h)

    g = conv.pconvfft(f,h)
    g2 = ia.pconv(f,h)

    print('g e g2 são iguais?: ', (abs(g - g2)<10E-4).all())

    get_ipython().magic('timeit pconvfft(h,f)')
    get_ipython().magic('timeit pconv(h,f)')


# In[ ]:

if testing:
    # Teste 2D
    f = mpimg.imread('/home/lotufo/ia898/data/gull.pgm')
    #h = np.ones((5,5))
    h = np.zeros(f.shape)
    h[::4,::4]=1

    g = conv.pconvfft(f,h)
    g2 = pconv(f,h)

    F = np.fft.fft2(f)

    print('g e g2 são iguais?: ', (abs(g - g2)<10E-4).all())

    get_ipython().magic('timeit pconvfft(f,h)')
    get_ipython().magic('timeit pconv(f,h)')

