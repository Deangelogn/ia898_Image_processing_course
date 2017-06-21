
# coding: utf-8

# # isccsym - Is Complex Conjugate Simmetric?

# Fazer uma função mais eficiente que a isdftsym abaixo:

# In[1]:

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import sys,os
path = os.path.abspath('/etc/jupyterhub/ia898_1s2017/d191122/Aula_6/')
if path not in sys.path:
    sys.path.append(path)
import ptrans as ptr


# In[2]:

import numpy as np

def isdftsym(F):

    if len(F.shape) == 1: F = F[np.newaxis,np.newaxis,:] 
    if len(F.shape) == 2: F = F[np.newaxis,:,:] 

    n,m,p = F.shape
    x,y,z = np.indices((n,m,p))

    Xnovo = np.mod(-1*x,n) 
    Ynovo = np.mod(-1*y,m)
    Znovo = np.mod(-1*z,p)  
     
    aux = np.conjugate(F[Xnovo,Ynovo,Znovo])
    
    return (abs(F-aux)<10E-4).all()



# In[18]:

# implementação oficial
def isccsym(F):
    H,W = F.shape
    
    U = (H//2)
    V = (W//2)
    
    pq = F[1:U+1,-V: ]   # primeiro quadrante
    sq = F[1:U+1, 1:V+1] # segundo quadrante
    tq = np.conjugate(np.flipud(np.fliplr(F[-U:,1:V+1]))) # terceiro quadrante
    qq = np.conjugate(np.flipud(np.fliplr(F[-U:,-V: ])))  # quarto quadrante
    
    # eixo vertical
    eV1 = F[0,1:V+1] 
    if(W%2==0):
        eV2 = np.conjugate(np.flipud(F[0,V:]))
    else:
        eV2 = np.conjugate(np.flipud(F[0,V+1:]))
        
    #eixo horizontal    
    eU1 = F[1:U+1,0]
    if(H%2==0):
        eU2 = np.conjugate(np.flipud(F[U:,0]))
    else:
        eU2 = np.conjugate(np.flipud(F[U+1:,0]))
        
    A1 = ((abs(pq-tq) + abs(sq-qq))<10E-4).all()
    A2 = ((abs(eV1-eV2))<10E-4).all()
    A3 = ((abs(eU1-eU2))<10E-4).all()
    A4 = ((abs(F[0,0]-np.conjugate(F[0,0])))<10E-4).all()
    
    return (A1 & A2 & A3 & A4)   


# In[4]:

#implementação alternativa
def isccsymAlt(F):
    H,W = F.shape    
    U = (H//2)
    V = (W//2)    
    fq = F[:U+1,:V+1]
    Ft = ptr.ptransfat(F,(-1,-1))   
    if (H%2==0):
        Ux = U-1
    else:
        Ux = U    
    if (W%2==0):
        Vy = V-1
    else:
        Vy = V   
    tq = np.conjugate(np.flipud(np.fliplr(Ft[Ux:,Vy:])))   
    sq = F[1:U+1,-V:]
    qq = np.conjugate(np.flipud(np.fliplr(F[-U:,1:V+1])))
    
    A1 = ((abs(fq-tq))<10E-4).all()
    A2 = ((abs(sq-qq))<10E-4).all()
    
    return A1 & A2


# In[5]:

testing = (__name__ == '__main__')

if testing:
    import sys,os
    get_ipython().system(" jupyter nbconvert --to 'python' isccsym")

    path = os.path.abspath('/etc/jupyterhub/ia898_1s2017/d191122/Aula_8/')
    if path not in sys.path:
        sys.path.append(path)
    import isccsym as ccs


# ## Testes para casos numéricos

# In[6]:

if testing:
    
    F = np.arange(4).reshape(2,2)
    print(F)
    
    G = ccs.isdftsym(F)
    G2 = ccs.isccsym(F)
    print("\nisdftsym:\n",G)
    print("\nisccsym:\n",G2)


# In[7]:

if testing:
    F2 = np.array([[2j,1j],
                   [1j,1j]])
    
    print(F2)
    G = ccs.isdftsym(F2)
    G2 = ccs.isccsym(F2)
    print("\nisdftsym:\n",G)
    print("\nisccsym:\n",G2)    


# In[8]:

if testing:
    F3 = np.array([[2j,1j],
                   [-1j,0]])
    
    print(F3)
    G =ccs.isdftsym(F3)
    G2 = ccs.isccsym(F3)
    print("\nisdftsym:\n",G)
    print("\nisccsym:\n",G2) 


# In[9]:

if testing:
    F4 = np.array([[2   ,2+1j,2-1j],
                   [1-1j,1+2j,2-2j],
                   [1+1j,2+2j,1-2j]])
    
    print(F4)
    G =ccs.isdftsym(F4)
    G2 = ccs.isccsym(F4)
    print("\nisdftsym:\n",G)
    print("\nisccsym:\n",G2) 


# In[19]:

if testing:
    F5 = np.array([[2   ,2+1j,2-1j],
                   [1-1j,1+2j,2-2j],
                   [1+1j,2+2j,2-2j],
                   [1+1j,2+2j,1-2j]])
    
    print(F5)
    G = ccs.isdftsym(F5)
    G2 = isccsym(F5)
    print("\nisdftsym:\n",G)
    print("\nisccsym:\n",G2) 


# ## Testando para verificar se funciona tanto para imagens com dimensões distintas, par e impar

# In[11]:

if testing:
    #imagem de dimensões pares
    f = mpimg.imread('/home/lotufo/ia898/data/cameraman.tif')
    print("H,W:\n",f.shape)
    
    F6 = np.fft.fft2(f) 
    G = ccs.isdftsym(F6)
    G2 = ccs.isccsym(F6)
    
    print("\nisdftsym:\n",G)
    print("\nisccsym:\n",G2) 


# In[12]:

if testing:
    #image de dimensões impares

    f2 = f[1:,1:] 
    print("H,W:\n",f2.shape)
    
    F7 = np.fft.fft2(f2) 
    G = ccs.isdftsym(F7)
    G2 = ccs.isccsym(F7)
    
    print("\nisdftsym:\n",G)
    print("\nisccsym:\n",G2)


# In[13]:

if testing:
    #dimensões alternadas
    f3 = f[:,1:] 
    print("H,W:\n",f3.shape)
    
    F8 = np.fft.fft2(f3) 
    G =isdftsym(F8)
    G2 = isccsym(F8)
    
    print("\nisdftsym:\n",G)
    print("\nisccsym:\n",G2)
    
    f4 = f[1:,:] 
    print("H,W:\n",f4.shape)
    
    F9 = np.fft.fft2(f4) 
    G = ccs.isdftsym(F9)
    G2 = ccs.isccsym(F9)
    
    print("\nisdftsym:\n",G)
    print("\nisccsym:\n",G2)


# ## Performance das implementações

# In[14]:

if testing:
    #Perfomance das implementações
    
    f1 = np.zeros((500,500))
    f2 = np.zeros((1001,987))
    f3 = np.zeros((2511,1800))
    f4 = np.zeros((3000,2777))
    
    F1 = np.fft.fft2(f1)
    F2 = np.fft.fft2(f2)
    F3 = np.fft.fft2(f3)
    F4 = np.fft.fft2(f4)

    
    print ("Imagem de dimensões:",F1.shape)
    print ("isccsym:\n")
    get_ipython().magic('timeit ccs.isccsym(F1)')

    print("isdftsym:\n")
    get_ipython().magic('timeit ccs.isdftsym(F1)')

    print ("isccsym Alternativo:\n")
    get_ipython().magic('timeit ccs.isccsymAlt(F1)')
    
    print ("------------------------------------------------------------")
    print ("Imagem de dimensões:",F2.shape)
    print ("isccsym:\n")
    get_ipython().magic('timeit ccs.isccsym(F2)')

    print("isdftsym:\n")
    get_ipython().magic('timeit ccs.isdftsym(F2)')

    print ("isccsym Alternativo:\n")
    get_ipython().magic('timeit ccs.isccsymAlt(F2)')
    
    print ("------------------------------------------------------------")
    print ("Imagem de dimensões:",F3.shape)
    print ("isccsym:\n")
    get_ipython().magic('timeit ccs.isccsym(F3)')

    print("isdftsym:\n")
    get_ipython().magic('timeit ccs.isdftsym(F3)')

    print ("isccsym Alternativo:\n")
    get_ipython().magic('timeit ccs.isccsymAlt(F3)')
    
    print ("------------------------------------------------------------")
    print ("Imagem de dimensões:",F4.shape)
    print ("isccsym:\n")
    get_ipython().magic('timeit ccs.isccsym(F4)')

    print("isdftsym:\n")
    get_ipython().magic('timeit ccs.isdftsym(F4)')

    print ("isccsym Alternativo:\n")
    get_ipython().magic('timeit ccs.isccsymAlt(F4)')
    

