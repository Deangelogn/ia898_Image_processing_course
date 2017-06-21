
# coding: utf-8

# # dftdecompose -  Illustrate the decomposition of the image in primitive 2-D waves
# 
# This demonstration illustrates the decomposition of a step function image into cossenoidal waves of increasing frequencies.

# In[1]:

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import sys,os    


# In[2]:

def dftdecompose(f,u):
    '''
    Inputs:
    f - image
    u - cossenoid sum level
    '''
    import numpy as np
    import matplotlib.pyplot as plt

    F = np.fft.fft2(f) # Transfomada de f
    H,W = f.shape
    Fsum = np.zeros_like(F) # Variável acumulativa de cossenoides
    Flast = np.zeros_like(F)
    for ui in range(0,u+1):
        Faux = np.zeros_like(F) 
        # Extraindo a cossenoide em u
        Faux[:,ui] = F[:,ui]
        Faux[:,W-ui-1] = F[:,W-ui-1]
        # Acumulando as cossenoides 
        Fsum += Faux 
        if ((ui+1) == u+1):
            Flast = Faux
            
    plt.figure(1,figsize=(12,12))                   
    plt.subplot(131)
    plt.imshow(f,cmap='gray')
    plt.title("imagem original")
    plt.subplot(132)
    plt.imshow(np.abs(np.fft.ifft2(Fsum)),cmap='gray')
    plt.title("somatório em u = {}".format(u))
    plt.subplot(133)
    plt.imshow(np.abs(np.fft.ifft2(Flast)),cmap='gray')
    plt.title("cossenoide em u = {}".format(u))  


# In[3]:

testing = (__name__ == '__main__')

if testing:
    import sys,os
    get_ipython().system(" jupyter nbconvert --to 'python' dftdecompose")

    path = os.path.abspath('/etc/jupyterhub/ia898_1s2017/d191122/Aula_7/')
    if path not in sys.path:
        sys.path.append(path)
    import dftdecompose as dftdecom


# In[4]:

if testing:
    # Processo interativo
    from ipywidgets import interact, interactive, fixed
    import ipywidgets as widgets
    import numpy as np

    f = 50 * np.ones((128,128))
    f[:,     : 32] = 200
    f[:,64+32:   ] = 200
    H,W = f.shape

    W_widget = widgets.IntSlider(min=0,max=(W//2)-1,step=1,value=0, continuous_update=False)
    interact(dftdecom.dftdecompose,f=fixed(f),u = W_widget);


# In[ ]:

if testing:
    f = 50 * np.ones((128,128))
    f[:,     : 32] = 200
    f[:,64+32:   ] = 200
    plt.imshow(f,cmap='gray')
    plt.colorbar()


# In[ ]:

if testing:
    import numpy as np
    import matplotlib.pyplot as plt

    f = 50 * np.ones((128,128))
    f[:,     : 32] = 200
    f[:,64+32:   ] = 200

    F = np.fft.fft2(f) # Transfomada de f
    H,W = f.shape

    Fsum = np.zeros_like(F) # Variável acumulativa de cossenoides

    plt.figure(figsize=[20,200])                    
    for u in range(0,W//2):
        Faux = np.zeros_like(F) 

        # Extraindo a cossenoide em u
        Faux[:,u] = F[:,u]
        Faux[:,W-u-1] = F[:,W-u-1]

        Fsum += Faux # Acumulando as cossenoides 

        plt.subplot(32,4,(u*2)+1)
        plt.imshow(np.abs(np.fft.ifft2(Fsum)),cmap='gray')
        plt.title("soma em u = {}".format(u))

        plt.subplot(32,4,(u*2)+2)
        plt.imshow(np.abs(np.fft.ifft2(Faux)),cmap='gray')
        plt.title("cossenoide de u = {}".format(u))

