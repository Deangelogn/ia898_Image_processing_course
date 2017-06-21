
# coding: utf-8

# # Normalização com percentil
# 
# Implementar a função de normalização com percentil, baseando-se no notebook:
# 
# - [Normalização com percentil](https://t-adessowiki.fee.unicamp.br/user/lotufo/notebooks/ia898_1s2017/ia898/deliver/Normalizacao_com_percentil.ipynb)

# In[29]:

get_ipython().magic('matplotlib inline')
def nperc_normalize(f, p):
    '''
    f : imagem de entrada, em nível de cinza
    p : float entre 0 e 1.0, indicando a percentagem de pixels escuros 
        claros para a normalização.
    '''
    import numpy as np
    
    pMin = np.round(255 * p)
    pMax = np.round(255 * (1 - p) )
    
    g = np.zeros(f.shape)
    g[f<=pMin] = 0
    g[f>=pMax] = 255
    
    h = (f>pMin) & (f<pMax)
    
    gMax = 255
    gMin = 0
    
    if(pMax != pMin):
        g[h] = ( (f[h] - pMin)/(pMax - pMin) ) * (gMax - gMin) + gMin    
    g = g.astype(np.uint8)
    
    return g


# In[20]:

# Teste numérico
import numpy as np

f = np.arange(256)
p = 0.05
print("configuração de entrada:\n",f)
output = nperc_normalize(f , p)
print("\nmin: ",np.round(255*0.05), "\nmax: ",np.round(255*0.95), "\n")
print("configuração de saída:\n",output)


# In[31]:

# Testando função para pencentil p = 50
p =0.5
print("configuração de entrada:\n",f)
output2 = nperc_normalize(f , p)
print("\nmin: ",np.round(255*0.05), "\nmax: ",np.round(255*0.95), "\n")
print("configuração de saída:\n",output2)


# In[32]:

# Testando função para pencentil p > 50
p =0.8
print("configuração de entrada:\n",f)
output3 = nperc_normalize(f , p)
print("\nmin: ",np.round(255*0.05), "\nmax: ",np.round(255*0.95), "\n")
print("configuração de saída:\n",output3)


# In[28]:

#Teste com imagem

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import sys,os
ia898path = os.path.abspath('/etc/jupyterhub/ia898_1s2017/')
if ia898path not in sys.path:
    sys.path.append(ia898path)
import ia898.src as ia

Img = mpimg.imread('../data/cameraman.tif')

p = 0.02
outputImg = nperc_normalize(Img , p)

#plot settings
#img
fig = plt.figure(1)
sp = fig.add_subplot(1,2,1)
plt.imshow(Img, cmap="gray")
sp.set_title('Imagem de entrada')
sp=fig.add_subplot(1,2,2)
plt.imshow(outputImg, cmap="gray")
sp.set_title('Imagem de saída')
#hist
plt.figure(2)
h = ia.histogram(Img)
plt.plot(h), plt.title('Histograma da imagem original')
plt.figure(3)
h2 = ia.histogram(outputImg)
plt.plot(h2), plt.title('Histograma com percentil de p=%.2f ' % (p))

