
# coding: utf-8

# # Função m0v1(grayImg)
# 
# ## Entradas:
# 
# - grayImg - imagem cinza com intensidades que variem entre 0 a 255.
# 
# ## Saída:
# 
# - retorna a imagem com média 0 e variância 1 
# 

# In[14]:

import numpy as np

def m0v1(grayImg):
    
    ## Função m0v1(grayImg)
    ##
    ## Entradas:
    ## - grayImg - imagem cinza com intensidades que variem entre 0 a 255.
    ## Saída:
    ## - retorna a imagem com média 0 e variância 1 
    
    # Obtendo a média a a variância da imagem oiginal.
    meanImg = np.mean(grayImg)
    stdImg = np.std(grayImg)
    
    # Deslocando média e variância para 0 e 1, respectivamente.
    grayImg = grayImg - meanImg
    grayImg = grayImg/stdImg
    
    return grayImg


# ## Validação

# In[17]:

# teste de validação

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

Img = mpimg.imread('./data/gull.pgm') #lendo imagem

print("média da imagem original: ", np.mean(Img))
print("variância da imagem original: ", np.var(Img))

m0v1Img = m0v1(Img) # obtendo imagem com média 0 e variância 

print("\nmédia da imagem de saída: ", np.mean(m0v1Img))
print("variância da imagem de saída: ", np.var(m0v1Img))

