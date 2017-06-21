
# coding: utf-8

# In[2]:

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys,os
ia898path = os.path.abspath('/home/lotufo/')
if ia898path not in sys.path:
    sys.path.append(ia898path)
import ia898.src as ia


# In[3]:

def conv_media(f,window):
    H,W = window
    
    #Tratando caso de quando a janela é maior que a imagem
    if (H > f.shape[0]):
        newRow = np.zeros((H - f.shape[0],f.shape[1]))
        f = np.vstack([f,newRow])
    
    if (W > f.shape[1]):
        newCol = np.zeros((f.shape[0],W - f.shape[0]))
        f = np.hstack([f,newCol])
                
    # Criando H-1 e W-1, linhas e colunas extras para realizar a convolução periódica corretamente          
    fc = np.zeros( ( f.shape[0]+(H),f.shape[1]+(W) ) )
    fc[H:,W:]=f
    
    # Ajuste de periodicidade da imagem
    fc[1:H,:] = fc[:-(H):-1,:][::-1,:]
    fc[:,1:W] = fc[:,:-(W):-1][:,::-1]
    fc[1:H,1:W] = fc[:-(H):-1,:-(W):-1][::-1,::-1]
    
    # Imagem integral
    sat = fc.cumsum(0).cumsum(1)
     
    return (sat[H:,W:] - sat[:-H,W:] - sat[H:,:-W] + sat[:-H,:-W])/(H*W)


# In[17]:

def conv_media2(f,window):
    H,W = window
    f1 = np.empty(((f.shape[0]+H-1),(f.shape[1]+W-1)))
    f1[(H-1):,(W-1):] = f
    if H>1:
        f1[:(H-1),(W-1):] = f[(f.shape[0]-(H-1)):,:]
    if W>1:
        f1[(H-1):,:(W-1)] = f[:,(f.shape[1]-(W-1)):]
    if (H>1)&(W>1):
        f1[:(H-1),:(W-1)] = f[(f.shape[0]-(H-1)):,(f.shape[1]-(W-1)):]
    sat = f1.cumsum(1).cumsum(0)
    A = np.zeros_like(sat)
    B = np.zeros_like(sat)
    C = np.zeros_like(sat)
    D = np.zeros_like(sat)
    A = sat
    B[:,W:] = ia.ptrans(sat,(0,W))[:,W:]
    C[H:,:] = ia.ptrans(sat,(H,0))[H:,:]
    D[H:,W:] = ia.ptrans(sat,(H,W))[H:,W:]
    g = A + D - C - B
    g = g[(H-1):,(W-1):]
    return g


# In[3]:

testing = (__name__ == '__main__')

if testing:
    import sys,os
    get_ipython().system(" jupyter nbconvert --to 'python' conv_media")

    path = os.path.abspath('/etc/jupyterhub/ia898_1s2017/d191122/Aula_12/')
    if path not in sys.path:
        sys.path.append(path)
    import conv_media as convm


# ## Teste numérico

# In[4]:

if testing:
    f = np.arange(1,17).reshape(4,4)
    H, W = (1,2)
    h = np.ones((H,W))/(H*W)

    g = convm.conv_media(f,h.shape)
    g2 = ia.pconv(f,h)

    print('g:\n',g)
    print('g2:\n',g2)


# In[5]:

if testing:
    f = np.arange(1,17).reshape(4,4)
    H, W = (2,2)
    h = np.ones((H,W))/(H*W)

    g = convm.conv_media(f,h.shape)
    g2 = ia.pconv(f,h)

    print('g:\n',g)
    print('g2:\n',g2)


# In[6]:

if testing:    
    f = np.arange(1,17).reshape(4,4)
    H, W = (1,5)
    h = np.ones((H,W))/(H*W)

    g = convm.conv_media(f,h.shape)
    g2 = ia.pconv(f,h)

    print('g:\n',g)
    print('g2:\n',g2)


# In[7]:

if testing:    
    f = np.arange(1,21).reshape(5,4)
    H, W = (2,2)
    h = np.ones((H,W))/(H*W)

    g = convm.conv_media(f,h.shape)
    g2 = ia.pconv(f,h)

    print('g:\n',g)
    print('g2:\n',g2)


# ## Teste com imagens

# In[8]:

if testing: 
    f = mpimg.imread('/home/lotufo/ia898/data/cameraman.tif')
    H, W = (3,3)
    h = np.ones((H,W))/(H*W)

    g = convm.conv_media(f,h.shape)
    g2 = ia.pconv(f,h)
       
    nb = ia.nbshow(3)
    nb.nbshow(f, 'Imagem original')
    nb.nbshow(ia.normalize(g), 'Imagem filtrada com conv_media' )
    nb.nbshow(ia.normalize(g2), 'Imagem filtrada com pconv')
    nb.nbshow()

    print('g e g2 são iguais? ', (abs(g-g2)<10E-4).all() )
    


# In[9]:

if testing: 
    
    f = mpimg.imread('/home/lotufo/ia898/data/cameraman.tif')
    H, W = (11,11)
    h = np.ones((H,W))/(H*W)

    g = convm.conv_media(f,h.shape)
    g2 = ia.pconv(f,h)
       
    nb = ia.nbshow(3)
    nb.nbshow(f, 'Imagem original')
    nb.nbshow(ia.normalize(g), 'Imagem filtrada com conv_media' )
    nb.nbshow(ia.normalize(g2), 'Imagem filtrada com pconv')
    nb.nbshow()

    print('g e g2 são iguais? ', (abs(g-g2)<10E-4).all() )


# ## Teste de perfomance

# In[10]:

if testing:
    path = os.path.abspath('/etc/jupyterhub/ia898_1s2017/d191122/Aula_10/')
    if path not in sys.path:
        sys.path.append(path)
    import pconvfft as conv
    
    f = mpimg.imread('/home/lotufo/ia898/data/cameraman.tif')
    H, W = (3,3)
    h = np.ones((H,W))/(H*W)
    
    print('h3x3:\n')
    print('-------------------------------------------\n')
    print('conv_media:\n')
    get_ipython().magic('timeit convm.conv_media(f,h.shape)')
    print('\n-------------------------------------------\n')
    print('pconv:\n')
    get_ipython().magic('timeit ia.pconv(f,h)')
    print('\n-------------------------------------------\n')
    print('pconvfft:\n')
    get_ipython().magic('timeit conv.pconvfft(f,h,0)')
    print('\n-------------------------------------------\n')
    
    h = np.ones((11,11))/(H*W)
    
    print('\n\n\nh11x11:\n')
    print('-------------------------------------------\n')
    print('conv_media:\n')
    get_ipython().magic('timeit convm.conv_media(f,h.shape)')
    print('\n-------------------------------------------\n')
    print('pconv:\n')
    get_ipython().magic('timeit ia.pconv(f,h)')
    print('\n-------------------------------------------\n')
    print('pconvfft:\n')
    get_ipython().magic('timeit conv.pconvfft(f,h,0)')
    print('\n-------------------------------------------\n')

