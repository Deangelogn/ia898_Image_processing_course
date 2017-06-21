
# coding: utf-8

# # isccsym - Is Complex Conjugate Simmetric?

# Implementar a função isccsym utilizando *slice* e que trate matrizes complexas. 
# Lembrar que basta testar metade do array, pois se F(a) == F(-a), não há necessidade de comparar F(-a) com F(a) novamente.
# 
# Testar se a função funciona com as imagens das listas ccsym.pkl

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



# In[3]:

def isccsym(F):
    # Faster implementation
    H,W = F.shape
    
    U = (H//2)
    V = (W//2)
              
    A1 = ((abs(F[1:U+1, 1:V+1]-np.conjugate(F[:-U-1:-1,:-V-1:-1])) + abs(F[-U:, 1:V+1]-np.conjugate(F[U:0:-1,:-V-1:-1])))<10E-4).all()
    A2 = ((abs(F[0,1:V+1]-np.conjugate(F[0,:-V-1:-1])))<10E-4).all()
    A3 = ((abs(F[1:U+1,0]-np.conjugate(F[:-U-1:-1,0])))<10E-4).all()
    A4 = ((abs(F[0,0]-np.conjugate(F[0,0])))<10E-4).all()
     
    return (A1 & A2 & A3 & A4)   


# In[4]:

def isccsym2(F):
    # Simple implementation
    H,W = F.shape
            
    A1 = ((abs(F[0,0]-np.conjugate(F[0,0])) )<10E-4).all()
    A2 = ((abs(F[0,1:]-np.conjugate(F[0,:0:-1])))<10E-4).all()
    A3 = ((abs(F[1:,0]-np.conjugate(F[:0:-1,0])  ))<10E-4).all()
    A4 = ((abs(F[1:,1:]-np.conjugate(F[:0:-1,:0:-1])))<10E-4).all()
    
    return (A1 & A2 & A3 & A4)   


# ## Testando com as imagens em ccsym.pkl

# In[5]:

import pickle

try:
    with open('/home/lotufo/ccsym.pkl','rb') as fhandle: 
        flist = pickle.load(fhandle)
except:
    print('arquivo não encontrado')
    
print(len(flist[0]),len(flist[1]))
    


# In[6]:

for img in flist[0]:
    print("isdftsym: ", isdftsym(img))
    print("isccsym: ", isccsym(img))


# In[7]:

for img in flist[1]:
    print("isdftsym: ",isdftsym(img))
    print("isccsym: ", isccsym(img))


# In[8]:

# Teste de performance das implementações
f4 = np.zeros((1000,1007))
F4 = np.fft.fft2(f4)
print('------------------------------\n')
print('isdftsym:\n')
get_ipython().magic('timeit isdftsym(F4)')
print('\n------------------------------\n')
print('isccsym (faster): \n')
get_ipython().magic('timeit isccsym(F4)')
print('\n------------------------------\n')
print('isccsym2 (simple): \n')
get_ipython().magic('timeit isccsym2(F4)')
print('\n------------------------------\n')

