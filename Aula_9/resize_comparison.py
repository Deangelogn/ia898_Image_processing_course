
# coding: utf-8

# ## resize_comparison
# 
# Comparação da implementação de ``resize`` ou similares em outros pacotes.
# 
# Utilizar no mínimo os pacotes scipy (PIL e o zoom) e skimage para analisar:
# - ampliação com interpolação linear
# - redução com interpolação linear
# 
# 1. Medir o tempo de processamento dessas implementações.
# 
# 2. Visualizar o espectro das imagens originais e interpoladas para inferir o tipo de implementação que é feito nestas implementações

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

import skimage.transform
import scipy

f = plt.imread('/home/lotufo/ia898/data/cameraman.tif')

H,W = f.shape
scale1 = 2
scale2 = 0.5

np.set_printoptions(precision=2)
nb = ia.nbshow(1)
nb.nbshow(f,'Imagem original')
nb.nbshow()


# ## skimage.transform.rescale

# In[4]:

# skimage.transform.rescale

skimage_rescale1 = skimage.transform.rescale(f,scale1)
skimage_rescale2 = skimage.transform.rescale(f,scale2)

nb = ia.nbshow(3)
nb.nbshow(ia.normalize(skimage_rescale2),'skimage_rescale x%s'%(scale2))
nb.nbshow(f,'Imagem original')
nb.nbshow(ia.normalize(skimage_rescale1),'skimage_rescale x%s'%(scale1))
nb.nbshow()

print("\nskimage x0.5:\n",skimage_rescale2[:8,:8])
print("f:\n",f[:8,:8])
print("\nskimage x2:\n",skimage_rescale1[:8,:8])


# ## skimage.transform.resize

# In[5]:

# skimage.transform.resize
skimage_resize1  = skimage.transform.resize(f,(H*scale1,W*scale1))
skimage_resize2  = skimage.transform.resize(f,((int)(H*scale2),(int)(W*scale2)))

nb = ia.nbshow(3)
nb.nbshow(ia.normalize(skimage_resize2),'skimage_resize x%s'%(scale2))
nb.nbshow(f,'Imagem original')
nb.nbshow(ia.normalize(skimage_resize1),'skimage_resize x%s'%(scale1))
nb.nbshow()

print("\nskimage x0.5:\n",skimage_resize2[:8,:8])
print("f:\n",f[:8,:8])
print("\nskimage x2:\n",skimage_resize1[:8,:8])


# ## scipy.misc.imresize

# In[6]:

# scipy.misc.imresize
scipy_imresize1 = scipy.misc.imresize(f,(H*scale1,W*scale1))
scipy_imresize2 = scipy.misc.imresize(f,((int)(H*scale2),(int)(W*scale2)))

nb = ia.nbshow(3)
nb.nbshow(scipy_imresize2,'scipy_imresize x%s'%(scale2))
nb.nbshow(f,'Imagem original')
nb.nbshow(scipy_imresize1,'scipy_imresize x%s'%(scale1))
nb.nbshow()

print("\nskimage x0.5:\n",scipy_imresize2[:8,:8])
print("f:\n",f[:8,:8])
print("\nskimage x2:\n",scipy_imresize1[:8,:8])


# ## scipy.ndimage.interpolation.zoom

# In[7]:

# scipy.ndimage.interpolation.zoom
scipy_zoom1 = scipy.ndimage.interpolation.zoom(f,scale1)
scipy_zoom2 = scipy.ndimage.interpolation.zoom(f,scale2)

nb = ia.nbshow(3)
nb.nbshow(scipy_zoom2,'scipy_zoom x%s'%(scale2))
nb.nbshow(f,'Imagem original')
nb.nbshow(scipy_zoom1,'scipy_zoom x%s'%(scale1))
nb.nbshow()

print("\nskimage x0.5:\n",scipy_zoom2[:8,:8])
print("f:\n",f[:8,:8])
print("\nskimage x2:\n",scipy_zoom1[:8,:8])


# In[8]:

print("\nskimage_resize e skimage_rescale utilizam o mesmo método de interpolação?\n", 
      (abs(skimage_resize1-skimage_rescale1)<10E-4).all())

print("\nscipy_imresize e scipy_zoom utilizam o mesmo método de interpolação?\n", 
      (abs(scipy_imresize1-scipy_zoom1)<10E-4).all())


# ## Teste de performance

# In[9]:

#Teste de performance das funções

print('imagem original: ',f.shape)
print('----------------------------------')
print('imagem duas vezes maior: ')
print('----------------------------------')
print('\nskimage_rescale1: ', skimage_rescale1.shape)
get_ipython().magic('timeit skimage.transform.rescale(f,scale1)')
print('\nskimage_resize1: ', skimage_resize1.shape)
get_ipython().magic('timeit skimage.transform.resize(f,(H*scale1,W*scale1))')
print('\nscipy_imresize1: ', scipy_imresize1.shape)
get_ipython().magic('timeit scipy.misc.imresize(f,(H*scale1,W*scale1))')
print('\nscipy_zoom1: ', scipy_zoom1.shape)
get_ipython().magic('timeit scipy.ndimage.interpolation.zoom(f,scale1)')

print('\n----------------------------------')
print('imagem duas vezes menor: ')
print('----------------------------------')
print('\nskimage_rescale2: ', skimage_rescale2.shape)
get_ipython().magic('timeit skimage.transform.rescale(f,scale2)')
print('\nskimage_resize2: ', skimage_resize2.shape)
get_ipython().magic('timeit skimage.transform.resize(f,((int)(H*scale2),(int)(W*scale2)))')
print('\nscipy_imresize2: ', scipy_imresize2.shape)
get_ipython().magic('timeit scipy.misc.imresize(f,((int)(H*scale2),(int)(W*scale2)))')
print('\nscipy_zoom2: ', scipy_zoom2.shape)
get_ipython().magic('timeit scipy.ndimage.interpolation.zoom(f,scale2)')


# ## Espectros das funções

# In[10]:

# Calculando espectro
F = np.fft.fft2(f)

FskiResc1 = np.fft.fft2(skimage_rescale1)
FskiResc2 = np.fft.fft2(skimage_rescale2)

FskiResz1 = np.fft.fft2(skimage_resize1)
FskiResz2 = np.fft.fft2(skimage_resize2)

FsciResz1 = np.fft.fft2(scipy_imresize1)
FsciResz2 = np.fft.fft2(scipy_imresize2)

FsciZoom1 = np.fft.fft2(scipy_zoom1)
FsciZoom2 = np.fft.fft2(scipy_zoom2)


# In[12]:

# Plotando espectros

nb = ia.nbshow(1)
#nb.nbshow(f, 'Imagem original')
nb.nbshow(ia.dftview(F), 'Espectro original')
nb.nbshow()


nb = ia.nbshow(2)

#nb.nbshow(ia.normalize(skimage_rescale1),'skiResc x%s'%(scale1))
#nb.nbshow(ia.normalize(skimage_resize1),'skiResz x%s'%(scale1))
#nb.nbshow(ia.normalize(scipy_imresize1),'sciResz x%s'%(scale1))
#nb.nbshow(ia.normalize(scipy_zoom1),'sciZoom x%s'%(scale1))

nb.nbshow(ia.dftview(FskiResc1),'skimage_rescale x%s'%(scale1))
nb.nbshow(ia.dftview(FskiResz1),'skimage_resize x%s'%(scale1))
nb.nbshow(ia.dftview(FsciResz1),'scipy_imresize x%s'%(scale1))
nb.nbshow(ia.dftview(FsciZoom1),'scipy_Zoom x%s'%(scale1))
nb.nbshow()

nb = ia.nbshow(4)

#nb.nbshow(ia.normalize(skimage_rescale2),'skiResc x%s'%(scale2))
#nb.nbshow(ia.normalize(skimage_resize2),'skiResz x%s'%(scale2))
#nb.nbshow(ia.normalize(scipy_imresize2),'sciResz x%s'%(scale2))
#nb.nbshow(ia.normalize(scipy_zoom2),'sciZoom x%s'%(scale2))

nb.nbshow(ia.dftview(FskiResc2),'skimage_rescale x%s'%(scale2))
nb.nbshow(ia.dftview(FskiResz2),'skimage_resize x%s'%(scale2))
nb.nbshow(ia.dftview(FsciResz2),'scipy_imresize x%s'%(scale2))
nb.nbshow(ia.dftview(FsciZoom2),'scipy_Zoom x%s'%(scale2))
nb.nbshow()

