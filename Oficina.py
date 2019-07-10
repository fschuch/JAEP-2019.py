#!/usr/bin/env python
# coding: utf-8

# <img src="JEAP.jpg" width="480">

# <h1>Sumário<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introdução" data-toc-modified-id="Introdução-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introdução</a></span><ul class="toc-item"><li><span><a href="#Sobre-o-autor" data-toc-modified-id="Sobre-o-autor-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Sobre o autor</a></span></li><li><span><a href="#Sobre-o-material" data-toc-modified-id="Sobre-o-material-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Sobre o material</a></span></li><li><span><a href="#Porque-Python?-ref" data-toc-modified-id="Porque-Python?-ref-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Porque Python? <a href="https://www.hostgator.com.br/blog/10-motivos-para-voce-aprender-python/" target="_blank">ref</a></a></span></li><li><span><a href="#Porque-Jupyter-Notebooks?" data-toc-modified-id="Porque-Jupyter-Notebooks?-1.4"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>Porque Jupyter Notebooks?</a></span></li><li><span><a href="#Material-Complementar" data-toc-modified-id="Material-Complementar-1.5"><span class="toc-item-num">1.5&nbsp;&nbsp;</span>Material Complementar</a></span></li></ul></li><li><span><a href="#Revisão" data-toc-modified-id="Revisão-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Revisão</a></span><ul class="toc-item"><li><span><a href="#Módulos" data-toc-modified-id="Módulos-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Módulos</a></span></li><li><span><a href="#Classes" data-toc-modified-id="Classes-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Classes</a></span></li><li><span><a href="#Dicionários" data-toc-modified-id="Dicionários-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Dicionários</a></span></li><li><span><a href="#Principais-Bibliotecas" data-toc-modified-id="Principais-Bibliotecas-2.4"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>Principais Bibliotecas</a></span><ul class="toc-item"><li><span><a href="#SciPy" data-toc-modified-id="SciPy-2.4.1"><span class="toc-item-num">2.4.1&nbsp;&nbsp;</span>SciPy</a></span></li><li><span><a href="#Numpy" data-toc-modified-id="Numpy-2.4.2"><span class="toc-item-num">2.4.2&nbsp;&nbsp;</span>Numpy</a></span></li><li><span><a href="#Exemplos" data-toc-modified-id="Exemplos-2.4.3"><span class="toc-item-num">2.4.3&nbsp;&nbsp;</span>Exemplos</a></span></li><li><span><a href="#Pandas" data-toc-modified-id="Pandas-2.4.4"><span class="toc-item-num">2.4.4&nbsp;&nbsp;</span>Pandas</a></span></li><li><span><a href="#Matplotlib" data-toc-modified-id="Matplotlib-2.4.5"><span class="toc-item-num">2.4.5&nbsp;&nbsp;</span>Matplotlib</a></span></li><li><span><a href="#Bokeh" data-toc-modified-id="Bokeh-2.4.6"><span class="toc-item-num">2.4.6&nbsp;&nbsp;</span>Bokeh</a></span></li></ul></li><li><span><a href="#Boas-práticas-em-programação" data-toc-modified-id="Boas-práticas-em-programação-2.5"><span class="toc-item-num">2.5&nbsp;&nbsp;</span>Boas práticas em programação</a></span></li><li><span><a href="#Fortran-vs.-Python" data-toc-modified-id="Fortran-vs.-Python-2.6"><span class="toc-item-num">2.6&nbsp;&nbsp;</span>Fortran vs. Python</a></span></li></ul></li><li><span><a href="#Exercícios-Resolvidos" data-toc-modified-id="Exercícios-Resolvidos-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Exercícios Resolvidos</a></span><ul class="toc-item"><li><span><a href="#Métodos-numéricos" data-toc-modified-id="Métodos-numéricos-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Métodos numéricos</a></span><ul class="toc-item"><li><span><a href="#Diferenciação" data-toc-modified-id="Diferenciação-3.1.1"><span class="toc-item-num">3.1.1&nbsp;&nbsp;</span>Diferenciação</a></span></li><li><span><a href="#Integração" data-toc-modified-id="Integração-3.1.2"><span class="toc-item-num">3.1.2&nbsp;&nbsp;</span>Integração</a></span></li></ul></li><li><span><a href="#Fenômenos-de-Transporte" data-toc-modified-id="Fenômenos-de-Transporte-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Fenômenos de Transporte</a></span><ul class="toc-item"><li><span><a href="#1D-Radiação-+-Convecção" data-toc-modified-id="1D-Radiação-+-Convecção-3.2.1"><span class="toc-item-num">3.2.1&nbsp;&nbsp;</span>1D Radiação + Convecção</a></span></li></ul></li><li><span><a href="#2D(t)" data-toc-modified-id="2D(t)-3.3"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>2D(t)</a></span><ul class="toc-item"><li><span><a href="#Convecção-/-Difusão" data-toc-modified-id="Convecção-/-Difusão-3.3.1"><span class="toc-item-num">3.3.1&nbsp;&nbsp;</span>Convecção / Difusão</a></span></li></ul></li><li><span><a href="#Vibrações-Mecânicas" data-toc-modified-id="Vibrações-Mecânicas-3.4"><span class="toc-item-num">3.4&nbsp;&nbsp;</span>Vibrações Mecânicas</a></span></li><li><span><a href="#Controle-e-Automação" data-toc-modified-id="Controle-e-Automação-3.5"><span class="toc-item-num">3.5&nbsp;&nbsp;</span>Controle e Automação</a></span></li><li><span><a href="#Engenharia-Econômica" data-toc-modified-id="Engenharia-Econômica-3.6"><span class="toc-item-num">3.6&nbsp;&nbsp;</span>Engenharia Econômica</a></span></li></ul></li><li><span><a href="#Exercícios-Propostos" data-toc-modified-id="Exercícios-Propostos-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Exercícios Propostos</a></span><ul class="toc-item"><li><span><a href="#Eletronica" data-toc-modified-id="Eletronica-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Eletronica</a></span></li><li><span><a href="#Resistência-dos-Materiais" data-toc-modified-id="Resistência-dos-Materiais-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Resistência dos Materiais</a></span></li></ul></li></ul></div>

# ## Introdução

# ### Sobre o autor
# 
# Possui graduação em Engenharia Mecânica pela Pontifícia Universidade Católica do Rio Grande do Sul (2013) e Mestrado em Engenharia e Tecnologia de Materiais pela mesma instituição. Atualmente atua como doutorando no Laboratório de Simulação de Escoamentos Turbulentos, Escola Politécnica da PUCRS. Possui experiencia em mecânica dos fluidos computacional, simulação numérica direta, simulação de grandes escalas, fenômenos de transporte, programação, programação paralela e métodos numéricos.
# 
# > **Felipe Nornberg Schuch**,<br>
# > Laboratório de Simulação de Escoamentos Turbulentos (LaSET),<br>
# > Escola Politécnica, Pontifícia Universidade Católica do Rio Grande do Sul.<br>
# > felipe.schuch@edu.pucrs.br

# ### Sobre o material

# ### Porque Python? [ref](https://www.hostgator.com.br/blog/10-motivos-para-voce-aprender-python/)
# 
# 1. Simples e fácil de aprender
# 2. Portátil e Extensível
# 3. Desenvolvimento Web
# 4. Inteligência Artificial
# 5. Computação Gráfica
# 6. Enquadramento de Testes
# 7. Big Data
# 8. Scripting e Automação
# 9. Ciência de Dados
# 10. Popularidade

# ### Porque Jupyter Notebooks?
# 
# * [Markdown quick reference](https://en.support.wordpress.com/markdown-quick-reference/)
# * [Jupyter tools to increase productivity](https://towardsdatascience.com/jupyter-tools-to-increase-productivity-7b3c6b90be09)
# * [LaTeX/Mathematics](https://en.wikibooks.org/wiki/LaTeX/Mathematics)

# ### Material Complementar
# 
# * [12 Steps to Navier-Stokes](https://github.com/barbagroup/CFDPython)
# * [An example machine learning notebook](https://nbviewer.jupyter.org/github/rhiever/Data-Analysis-and-Machine-Learning-Projects/blob/master/example-data-science-notebook/Example%2520Machine%2520Learning%2520Notebook.ipynb)
# * [Mythbusters Demo GPU versus CPU](https://www.youtube.com/watch?v=-P28LKWTzrI)

# ## Revisão

# In[1]:


'''
Isso é um comentário
'''

print("Olá mundo")

# Isso também é um comentário


# In[48]:


# Declarando variáveis
i = 5        #inteiro
f = 6.7      #ponto flutuante
s = 'abcd'   #string
c = 5.0 + 6j #complexo


# In[38]:


i = i + 1    #acumulador 
i


# In[44]:


i += 1       #forma alternativa para acumulador
i


# In[42]:


#laço de zero a 4
for i in range(5):
    print(i)


# In[47]:


#teste lógico
if i == 4:
    print('i é igual a 4')
else:
    print('i não é igual a 4, i é igual a '+str(i))


# ### Módulos
# 
# ### Classes
# 
# ### Dicionários
# 
# * [Dictionaries in Python](https://realpython.com/python-dicts/)

# ### Principais Bibliotecas

# ####  SciPy
# 
# ![SciPy](https://www.scipy.org/_static/images/scipy_med.png "SciPy")
# 
# Ferramentas de computação científica para Python. SciPy refere-se a várias entidades relacionadas, mas distintas:
# 
# * O ecossistema SciPy, uma coleção de software de código aberto para computação científica em Python.
# * A comunidade de pessoas que usam e desenvolvem essa biblioteca.
# * Várias conferências dedicadas à computação científica em Python - SciPy, EuroSciPy e SciPy.in.
# * Fazem parte da família os pacotes, que serão melhor descritos a seguir:
#   * Numpy
#   * Matplotlib
#   * Sympy
#   * IPython
#   * Pandas
# * Além disso, a própria biblioteca SciPy, um componente do conjunto SciPy, fornecendo muitas rotinas numéricas:
#   * Funções especiais
#   * Integração numérica
#   * Diferenciação numérica
#   * Otimização
#   * Interpolação
#   * Transformada de Fourier
#   * Processamento de sinal
#   * Algebra linear e Algebra linear esparsa
#   * Problema de autovalor esparso com ARPACK
#   * Algoritmos e estruturas de dados espaciais
#   * Estatistica
#   * Processamento de imagem multidimensional
#   * I/O de arquivos
#   
# Material complementar:
# * [SciPy](https://www.scipy.org/)
# * [Getting Started](https://www.scipy.org/getting-started.html)
# * [Scipy Lecture Notes](http://scipy-lectures.org/index.html)

# ####  Numpy
# 
# ![Numpy](https://www.scipy.org/_static/images/numpylogo_med.png "Numpy")
# 
# Numpy é um pacote fundamental para a computação científica em Python. Entre outras coisas, destaca-se:
# * Objetos em arranjos N-dimensionais
# * Funções sofisticadas
# * Ferramentas para integrar código C/C++ e Fortran
# * Conveniente álgebra linear, transformada de Fourier e capacidade de números aleatórios
# 
# Além de seus usos científicos óbvios, o NumPy também pode ser usado como um contêiner multidimensional eficiente de dados genéricos. Tipos de dados arbitrários podem ser definidos. Isso permite que o NumPy integre-se de forma fácil e rápida a uma ampla variedade de bancos de dados.
# 
# #### Exemplos

# In[2]:


import numpy as np # Importando a biblioteca numpy e definindo-a com o codnome de np


# In[3]:


print(np.arange.__doc__) # É sempre possível checar a documentação de uma dada função


# In[4]:


a = np.arange(15).reshape(3, 5) # Criando um arranjo com 15 elementos e o redimensionando para o formato 3x5


# In[5]:


a # Escrevendo a


# In[6]:


a.shape # Verificando as dimensões do arranjo


# In[7]:


a.ndim # O número de dimensões


# In[8]:


a.dtype.name # Classificação quando ao tipo dos elementos


# In[9]:


a.itemsize # Tamanho em bytes de cada elemento


# In[10]:


a.size # Número total de elementos no arranjo


# In[11]:


type(a)


# In[12]:


# outras funções que merecem destaque:
for f in [np.zeros, np.zeros_like, np.ones, np.linspace]:
    print('=============== '+f.__name__+' ===============\n')
    print(f.__doc__+'\n')


# * [NumPy](https://www.numpy.org/)
# * [Quickstart tutorial](https://www.numpy.org/devdocs/user/quickstart.html)

# ####  Pandas
# 
# ![Pandas](https://www.scipy.org/_static/images/pandas_badge2.jpg "Pandas")
# 
# ####  Matplotlib
# 
# ![Matplotlib](https://www.scipy.org/_static/images/matplotlib_med.png "Matplotlib")
# 
# * [Style sheets reference](https://matplotlib.org/3.1.0/gallery/style_sheets/style_sheets_reference.html)
# * [Gallery](https://matplotlib.org/3.1.0/gallery/index.html)
# 
# ####  Bokeh
# 
#  ![Bokeh](https://bokeh.pydata.org/en/latest/_static/images/logo.png "Bokeh")
# 
# O Bokeh é uma biblioteca de visualização interativa para Python que permite uma apresentação visual de dados bonita e significativa em navegadores modernos. Com o Bokeh, você pode criar, de maneira rápida e fácil, plotagens interativas, painéis e aplicativos de dados.
# 
# O Bokeh fornece uma maneira elegante e concisa de construir gráficos versáteis e, ao mesmo tempo, oferecer interatividade de alto desempenho para conjuntos de dados grandes ou em fluxo.
# 
# * [Tutorial](https://mybinder.org/v2/gh/bokeh/bokeh-notebooks/master?filepath=tutorial%2F00%20-%20Introduction%20and%20Setup.ipynb)
# * [Gallery](https://bokeh.pydata.org/en/latest/docs/gallery.html)

# ### Boas práticas em programação

# ### Fortran vs. Python

# É preciso criar o arquivo `escreve_sin_cos.f90`
# ```fortran
# program escreve_sin_cos
# 
#     implicit none
#     
#     !Declarar variáveis
#     integer :: i, nx
#     real(4) :: xi, xf, dx
#     real(4), allocatable, dimension(:) :: x, sinx, cosx
# 
#     nx=200
#     xi = 0.
#     xf = 3.14
# 
#     !Alocar variáveis
#     allocate(x(nx), sinx(nx), cosx(nx))
# 
#     !Calcular
#     dx = (xi-xf)/(nx-1)
#     do i=1, nx
#         x(i) = real(i-1,4)*dx
#         sinx(i) = sin(x(i))
#         cosx(i) = cos(x(i))
#     end do
# 
#     !Escrever em disco e abrir em outro
#     !programa para gerar imagens
#     open (1, file = "teste.csv")
#     do i=1, nx
#         write(1,*) x(i), sinx(i), cosx(i)
#     end do
#     close(1)
#     
#     !Encerrar programa
# end program escreve_sin_cos
# ```
# 
# Além de compilar
# 
# `gfortran -o teste escreve_sin_cos.f90`,
# 
# executar:
# 
# `./teste`,
# 
# e ainda processar o arquivo de saída `teste.csv` em outro software.

# In[13]:


x = np.linspace(0., 2*np.pi, num=200, endpoint=True)

sinx = np.sin(x)
cosx = np.cos(x)

plt.plot(x, sinx, label='sin')
plt.plot(x, cosx, label='sin')

plt.legend()
plt.show()


# In[ ]:


x = np.linspace(0., 2*np.pi, num=200, endpoint=True)

for f in [np.sin, np.cos]:
    plt.plot(x, f(x), label=f.__name__)
    
plt.legend()
plt.show()


# In[ ]:


del x, f


# ## Exercícios Resolvidos

# In[ ]:


import numpy as np
import scipy as sp
import sympy as sm
import bokeh as bk
import pandas as pd
import matplotlib.pyplot as plt

sm.init_printing()


# ### Métodos numéricos

# #### Diferenciação
# 
# \begin{equation}
#     f_i' = \dfrac{f_{i+1}-f_{i-1}}{2\Delta x}
# \end{equation}
# 
# \begin{equation}
# \begin{split}
# \begin{bmatrix} f'_{0} \\ f'_{1} \\ \vdots \\ f'_{i} \\ \vdots \\ f'_{n-2}\\ f'_{n-1}\end{bmatrix} =
# \underbrace{
# \frac{1}{2 \Delta x}
# \begin{bmatrix}
# 0 & 1 & & & & & -1 \\
# -1 & 0 & 1 & & \\
#  & \ddots & \ddots & \ddots & & & \\
# & & -1 & 0 & 1 & \\
# & & & \ddots & \ddots & \ddots & \\
# & & & & -1 & 0 & 1\\
# 1 & & & & & -1 & 0
# \end{bmatrix}
# }_{D_x = \text{ Operador diferencial de primeira ordem}}
# \begin{bmatrix} f_{0} \\ f_{1} \\ \vdots \\ f_{i} \\ \vdots \\ f_{n-2}\\ f_{n-1}\end{bmatrix}
# \\
# \mbox{com $2\leq i \leq n -3$ e $f_0 = f_n$}
# \end{split}
# \label{eq.dxx_matrix}
# \end{equation}
# 
# \begin{equation}
#     f' = D_x f
# \end{equation}

# \begin{equation}
#     f_i'' = \dfrac{f_{i+1} - 2 f_{i} + f_{i-1}}{(\Delta x)^2}
# \end{equation}
# 
# \begin{equation}
# \begin{split}
# \begin{bmatrix} f''_{0} \\ f''_{1} \\ \vdots \\ f''_{i} \\ \vdots \\ f''_{n-2}\\ f''_{n-1}\end{bmatrix} =
# \underbrace{
# \frac{1}{(\Delta x)^2}
# \begin{bmatrix}
# -2 & 1 & & & & & 1\\
# 1 & -2 & 1 & & \\
#  & \ddots & \ddots & \ddots & & & \\
# & & 1 & -2 & 1 & \\
# & & & \ddots & \ddots & \ddots & \\
# & & & & 1 & -2 & 1\\
# 1 & & & & & 1 & -2
# \end{bmatrix}
# }_{D_x^2 = \text{ Operador diferencial de segunda ordem}}
# \begin{bmatrix} f_{0} \\ f_{1} \\ \vdots \\ f_{i} \\ \vdots \\ f_{n-2}\\ f_{n-1}\end{bmatrix}
# \\
# \mbox{com $2\leq i \leq n -3$ e $f_0 = f_n$}
# \end{split}
# \label{eq.dxx_matrix}
# \end{equation}
# 
# \begin{equation}
#     f'' = D_x^2 f
# \end{equation}

# In[ ]:


import scipy.sparse as sps

x = np.linspace(0., 2*np.pi, num=200, endpoint=False)

dx = (x[1]-x[0])

#Operador diferencial de primeira ordem
Dx = sps.diags([-1., 0., 1.],
              offsets=[-1, 0, 1],
              shape=(x.size,x.size)
              ).toarray()

#Operador diferencial de segunda ordem
Dx2 = sps.diags([1., -2., 1.],
               offsets=[-1, 0, 1],
               shape=(x.size,x.size)
               ).toarray()

Dx /= 2*dx
Dx2 /= dx**2.

#Condições de contorno Periódicas
for D in [Dx, Dx2]:
    D[0,-1] = D[1,0]
    D[-1,0] = D[-2,-1]


# In[ ]:


f = np.sin(x)

plt.plot(x, f, label="f(x)")
plt.plot(x, Dx.dot(f), label="f'(x)")
plt.plot(x, Dx2.dot(f), label="f''(x)")

plt.legend()
plt.show()


# #### Integração
# 
# 

# ### Fenômenos de Transporte

# #### 1D Radiação + Convecção

# In[ ]:


k, t1, t2, l, epsi, sigma, tviz, too, h = sm.symbols("k t_1 t_2 L \epsilon \sigma t_{viz} t_\infty h")


# In[ ]:


eq1 = sm.Eq(k*(t1-t2)/l,epsi*sigma*(t2**4.-tviz**4.)+h*(t2-too))
eq1


# In[ ]:


print(eq1.subs.__doc__)


# In[ ]:


dic = {k: 1., t1: 0., l: 1., epsi: 1., sigma: 1., tviz: 1., too: 1., h: 1.}
eq1.subs(dic)


# In[ ]:


print(sm.solve.__doc__)


# In[ ]:


sol = sm.solve(eq1.subs(dic), t2)


# In[ ]:


sol


# In[ ]:


for val in sol:
    if val.is_real:
        if val.is_positive:
            print(val)  


# In[ ]:


del k, t1, t2, l, epsi, sigma, tviz, too, h, eq1, dic, sol


# ### 2D(t)
# 
# Uma placa de alumínio de $40cm \times 40cm$ inicialmente possui temperatura em toda a sua extensão igual a $0^oC$. Instantaneamente , suas bordas são levadas às temperaturas de $75^oC$ em $x=0$; $50^oC$ em $x=40cm$; $0^oC$ em $y=0$ e $100^oC$ em $y=40$. A difusividade térmica do alumínio é $0,835cm^2/s$. Escolha um $\Delta t$ adequado com o critério de estabilidade para formulação explícita e calcule a evolução da temperatura para a posição central da placa até o tempo de $400s$. Para o tempo de $200s$ apresente o perfil de temperatura em todos os pontos discretos do domínio, usando $\Delta x=4cm$., dt=4
# 
# Equação bidimensional:
# 
# \begin{equation}
# \alpha \left( \dfrac{\partial ^2 T}{\partial x^2} + \dfrac{\partial ^2 T}{\partial y^2} \right) =\dfrac{\partial T}{\partial t}
# \end{equation}
# 
# Discretizando com a derivada segunda numa representação por diferença central e a derivada primeira com diferença ascendente:
# 
# \begin{equation}
# \dfrac{T^{n+1}_{l,j}-T^{n}_{l,j}}{\Delta t}=\alpha \left[ \dfrac{T^{n}_{l-1,j}-2T^{n}_{l,j}+T^{n}_{l+1,j}}{(\Delta x)^2} +\dfrac{T^{n}_{l,j-1}-2T^{n}_{l,j}+T^{n}_{l,j+1}}{(\Delta y)^2}  \right]
# \end{equation}

# In[ ]:


x = np.linspace(0., 40., num=11, endpoint=True)
y = np.linspace(0., 40., num=11, endpoint=True)
t = np.linspace(0., 400., num=101, endpoint=True)

a =0.835

T = np.zeros((x.size,y.size,t.size))

T[0,:,:] = 75.
T[-1,:,:] = 50.
T[:,0,:] = 0.
T[:,-1,:] = 100.


# In[ ]:


dt = t[1]-t[0]
dx2 = (x[1]-x[0])**2.
dy2 = (y[1]-y[0])**2.
for n in range(t.size-1):
    for i in range(1,x.size-1):
        for j in range(1,y.size-1):
            T[i,j,n+1] = dt*a*((T[i-1,j,n]-2.*T[i,j,n]+T[i+1,j,n])/dx2+(T[i,j-1,n]-2.*T[i,j,n]+T[i,j+1,n])/dy2)+T[i,j,n]


# In[ ]:


#Adicionar subplot
fig, ax = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True)

for i, n in enumerate(T[:,:,::25].T):
    ax[i].imshow(n)
    ax[i].set_title('t={}'.format(t[i*25]))
    
plt.show()


# In[ ]:


plt.plot(t, T[5,5,:])
plt.title('Evolução da temperatura no centro da placa')
plt.xlabel('t')
plt.ylabel('T')
plt.show()


# In[ ]:


del T, x, y, t, a, dx2, dy2, dt, i, n


# #### Convecção / Difusão
# 
# Resolver a EDP:
# \begin{equation}
#     \dfrac{\partial T}{\partial t}+u\dfrac{\partial T}{\partial x}=\alpha \dfrac{\partial^2 T}{\partial x^2} \quad 0\leq x \leq 1 ; 0\leq t \leq 8
# \end{equation}
# Condições de contorno:
# \begin{equation}
#     T(0,t)=T(1,t)=0
# \end{equation}
# Condição inicial:
# \begin{equation}
#     T(x,0)=  \left\{ \begin{tabular}{c l}
# 0 & se \\
# 0 & se \end{tabular}
# \right.
# \end{equation}
# 
# Discretizando com a derivada segunda numa representação por diferença central e a derivada primeira com diferença ascendente:
# \begin{equation}
# \dfrac{T_{i,k+1}-T_{i,k}}{\Delta t}=\alpha \dfrac{T_{i-1,k}-2T_{i,k}+T_{i+1,k}}{(\Delta x)^2} -u\dfrac{T_{i+1,k}-T_{i-1,k}}{2\Delta x}
# \end{equation}

# O problema pode ser escrito na forma matricial como:
# \begin{equation}
# \begin{split}
# \begin{bmatrix} T_{0,k+1} \\ T_{1,k+1} \\ \vdots \\ T_{i,k+1} \\ \vdots \\ T_{n-2,k+1}\\ T_{n-1,k+1}\end{bmatrix} =
# \begin{bmatrix} T_{0,k} \\ T_{1,k} \\ \vdots \\ T_{i,k} \\ \vdots \\ T_{n-2,k}\\ T_{n-1,k}\end{bmatrix} +
# \frac{\alpha \Delta t}{(\Delta x)^2}
# \begin{bmatrix}
# 0 & 0 & & & & & \\
# 1 & -2 & 1 & & \\
#  & \ddots & \ddots & \ddots & & & \\
# & & 1 & -2 & 1 & \\
# & & & \ddots & \ddots & \ddots & \\
# & & & & 1 & -2 & 1\\
#  & & & & & 0 & 0
# \end{bmatrix}
# \begin{bmatrix} T_{0,k} \\ T_{1,k} \\ \vdots \\ T_{i,k} \\ \vdots \\ T_{n-2,k}\\ T_{n-1,k}\end{bmatrix} -
# \frac{u \Delta t}{2\Delta x}
# \begin{bmatrix}
# 0 & 0 & & & & & \\
# -1 & 0 & 1 & & \\
#  & \ddots & \ddots & \ddots & & & \\
# & & -1 & 0 & 1 & \\
# & & & \ddots & \ddots & \ddots & \\
# & & & & -1 & 0 & 1\\
#  & & & & & 0 & 0
# \end{bmatrix}
# \begin{bmatrix} T_{0,k} \\ T_{1,k} \\ \vdots \\ T_{i,k} \\ \vdots \\ T_{n-2,k}\\ T_{n-1,k}\end{bmatrix}
# \\
# \mbox{com $2\leq i \leq n -3$,}
# \end{split}
# \label{eq.dxx_matrix}
# \end{equation}

# ou simplesmente:
# \begin{equation}
#    T_{k+1} = T_{k} + \underbrace{ \Delta t \left( \alpha D_x - u D_x^2 \right)}_{A} T_{k},
# \end{equation}
# onde $D_x$ e $D_x^2$ são os operadores diferenciais de primeira e segunda ordem, respectivamente.

# In[ ]:


import scipy.sparse as sps

x = np.linspace(0., 1, num=101, endpoint=True)
t = np.linspace(0., 8., num=8001, endpoint=True)

dx = (x[-1]-x[0])/(x.size-1)
dt = (t[-1]-t[0])/(t.size-1)

#Operador diferencial de primeira ordem
Dx = sps.diags([-1., 0., 1.],
              offsets=[-1, 0, 1],
              shape=(x.size,x.size)
              ).toarray()

#Operador diferencial de segunda ordem
Dx2 = sps.diags([1., -2., 1.],
              offsets=[-1, 0, 1],
              shape=(x.size,x.size)
               ).toarray()

#Condições de contorno
for D in [Dx, Dx2]:
    D[0,:] = 0.
    D[-1,:] = 0.


# In[ ]:


def convdiff(alpha, u):

    #Condição inicial
    T = np.zeros((x.size))
    for i, ival in enumerate(x):
        if ival > 0.2:
            break
        T[i] = 1. - (10. * ival - 1)**2.

    A = dt*(alpha*Dx2/(dx)**2.-u*Dx/(2.*dx))
    
    return A, T


# In[ ]:


#Parâmetro de visualização: a cada quantos passos de tempo se deve graficar os resultados
visu = 2000. 

alpha, u = 0.001, 0.08

A, T = convdiff(alpha, u)

for n in range(t.size):
    T += A.dot(T)
    if n % visu == 0:
        plt.plot(x, T, label='t={}'.format(t[n]))

plt.xlabel('x')
plt.ylabel('T(x)')
plt.legend()
plt.show()


# In[ ]:


fig, ax = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True)

for i, alpha in enumerate([0., 0.0005, 0.001]):
    for j, u in enumerate([0., 0.04, 0.08]):
        A, T = convdiff(alpha, u)
        for n in range(t.size):
            T += A.dot(T)
            if n % visu == 0:
                ax[i,j].plot(x, T, label='t={}'.format(t[n]))
                ax[-1,j].set_xlabel(r'$u = {}$'.format(u))
                ax[i,0].set_ylabel(r'$\alpha = {}$'.format(alpha))

plt.show()


# In[ ]:


del x, t, dx, dt, D, Dx, Dx2, visu, alpha, u, A, T
plt.close('all')


# ### Vibrações Mecânicas

# ### Controle e Automação

# ### Engenharia Econômica

# ## Exercícios Propostos

# ### Eletronica

# <img src="circuit.png" width="480">

# Pela primeira lei de Kirchhoff:
# \begin{equation}
#     I_1 + I_2 - I_3 = 0
# \end{equation}
# 
# Pela segunda lei de Kirchhoff:
# \begin{equation}
# R_4 I_1 + R_3 I_3 + R_1 I_1 = V_1
# \end{equation}
# \begin{equation}
# R_5 I_2 + R_3 I_3 + R_2 I_2 = V_2
# \end{equation}

# In[ ]:


r1, r2, r3, r4, r5 = 10., 22., 47., 22., 10.
v1, v2 = 9., 9.

A = [[1., 1., -1.],
     [r1+r4, 0., r3],
     [0., r2+r5, r3]]

B = [0., v1, v2]


# In[ ]:


I = np.linalg.solve(A,B)


# In[ ]:


for i, ival in enumerate(I):
    print('I{} = {}'.format(i, ival))


# ### Resistência dos Materiais

# <img src="trelica.png" width="640">

# \begin{equation}
#   F_{1x} - F_{12} - \cos(30^o) F_{15} = 0\\
#   F_{1y} - \sin(30^o) F_{15} = 0 \\
#   F_{12} - F_{23} + \cos(60^o) F_{25} - \cos(60^o) F_{27} = 0 \\
#   - \sin(60^o) F_{25} - \sin(60^o) F_{27} = 0 \\
#   F_{23} - F_{34} + \cos(60^o) F_{37} - \cos(60^o) F_{36} = 0 \\
#   - \sin(60^o) F_{37} - \sin(60^o) F_{36} = 0 \\
#   F_{34} + \cos(30^o) F_{46} = 0 \\
#   F_{4y} - \sin(30^o) F_{46} = 0 \\
#   \cos(30^o) F_{15} - \cos(60^o) F_{25} - \cos(30^o) F_{57} = 0 \\
#   \sin(30^o) F_{15} + \sin(60^o) F_{25} - \sin(30^o) F_{57} - F_{5} = 0 \\
#    - \cos(30^o) F_{46} + \cos(60^o) F_{36} + \cos(30^o) F_{67} = 0 \\
#   \sin(30^o) F_{46} + \sin(60^o) F_{36} - \sin(30^o) F_{67} - F_{6} = 0 \\
#   \cos(30^o) F_{57} + \cos(60^o) F_{27} - \cos(30^o) F_{67} - \cos(60^o) F_{37} = 0 \\
#   \sin(30^o) F_{57} + \sin(60^o) F_{27} + \sin(30^o) F_{67} + \sin(60^o) F_{37} - F_7 = 0
# \end{equation}

# \begin{equation}
# \begin{split}
# \begin{bmatrix}
# 1	&	0	&	0	&	-\cos	&	-1	&	0	&	0	&	0	&	0	&	0	&	0	&	0	&	0	&	0	\\
# 0	&	1	&	0	&	-\sin	&	0	&	0	&	0	&	0	&	0	&	0	&	0	&	0	&	0	&	0	\\
# 0	&	0	&	0	&	0	&	1	&	-1	&	\sin	&	-\sin	&	0	&	0	&	0	&	0	&	0	&	0	\\
# 0	&	0	&	0	&	0	&	0	&	0	&	-\cos	&	-\cos	&	0	&	0	&	0	&	0	&	0	&	0	\\
# 0	&	0	&	0	&	0	&	0	&	1	&	0	&	0	&	-1	&	-\sin	&	\sin	&	0	&	0	&	0	\\
# 0	&	0	&	0	&	0	&	0	&	0	&	0	&	0	&	0	&	-\cos	&	-\cos	&	0	&	0	&	0	\\
# 0	&	0	&	0	&	0	&	0	&	0	&	0	&	0	&	1	&	0	&	0	&	\cos	&	0	&	0	\\
# 0	&	0	&	1	&	0	&	0	&	0	&	0	&	0	&	0	&	0	&	0	&	-\sin	&	0	&	0	\\
# 0	&	0	&	0	&	\cos	&	0	&	0	&	-\sin	&	0	&	0	&	0	&	0	&	0	&	-\cos	&	0	\\
# 0	&	0	&	0	&	\sin	&	0	&	0	&	\cos	&	0	&	0	&	0	&	0	&	0	&	-\sin	&	0	\\
# 0	&	0	&	0	&	0	&	0	&	0	&	0	&	0	&	0	&	\sin	&	0	&	-\cos	&	0	&	\cos	\\
# 0	&	0	&	0	&	0	&	0	&	0	&	0	&	0	&	0	&	\cos	&	0	&	\sin	&	0	&	-\sin	\\
# 0	&	0	&	0	&	0	&	0	&	0	&	0	&	\sin	&	0	&	0	&	-\sin	&	0	&	\cos	&	-\cos	\\
# 0	&	0	&	0	&	0	&	0	&	0	&	0	&	\cos	&	0	&	0	&	\cos	&	0	&	\sin	&	\sin
# \end{bmatrix}
# \begin{bmatrix}
# F_{1x} \\ F_{1y} \\ F_{4y} \\ F_{15} \\ F_{12} \\ F_{23} \\ F_{25} \\ F_{27} \\ F_{34} \\ F_{36} \\ F_{37} \\ F_{46} \\ F_{57} \\ F_{67}
# \end{bmatrix}
# =
# \begin{bmatrix} 0 \\ 0 \\ 0 \\ 0 \\ 0 \\ 0 \\ 0 \\ 0 \\ 0 \\ F_5 \\ 0 \\ F_6 \\ 0 \\ F_7 \end{bmatrix}
# \end{split}
# \end{equation}

# In[ ]:


angle = np.deg2rad(30.)
F5, F6, F6 = 1., 1., 2.
cos, sin = np.cos(angle), np.sin(angle)


# In[ ]:


A = np.array([[1.,0.,0.,-cos,-1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
             [0.,1.,0.,-sin,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
             [0.,0.,0.,0.,1.,-1.,sin,-sin,0.,0.,0.,0.,0.,0.],
             [0.,0.,0.,0.,0.,0.,-cos,-cos,0.,0.,0.,0.,0.,0.],
             [0.,0.,0.,0.,0.,1.,0.,0.,-1.,-sin,sin,0.,0.,0.],
             [0.,0.,0.,0.,0.,0.,0.,0.,0.,-cos,-cos,0.,0.,0.],
             [0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,cos,0.,0.],
             [0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,-sin,0.,0.],
             [0.,0.,0.,cos,0.,0.,-sin,0.,0.,0.,0.,0.,-cos,0.],
             [0.,0.,0.,sin,0.,0.,cos,0.,0.,0.,0.,0.,-sin,0.],
             [0.,0.,0.,0.,0.,0.,0.,0.,0.,sin,0.,-cos,0.,cos],
             [0.,0.,0.,0.,0.,0.,0.,0.,0.,cos,0.,sin,0.,-sin],
             [0.,0.,0.,0.,0.,0.,0.,sin,0.,0.,-sin,0.,cos,-cos],
             [0.,0.,0.,0.,0.,0.,0.,cos,0.,0.,cos,0.,sin,sin]])

B = np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.,F5,0.,F6,0.,F7])


# In[ ]:


I = np.linalg.solve(A,B)
#from scipy.sparse.linalg import lsqr
#I = lsqr(A,B)[0]


# In[ ]:


for i, key in enumerate(['F1x','F1y','F4y','F15','F12','F23','F25','F27','F34','F36','F37','F46','F57','F67']):
    print(i,key,I[i])


# In[ ]:


l = 2. #[m]
angle = np.deg2rad(30.)

#T=O/A

points = {'1': [0., 0.],
          '2': [l, 0.],
          '3': [2.*l, 0.],
          '4': [3.*l, 0.],
          '5': [.75*l, .75*l*np.tan(angle)],
          '6': [2.25*l, .75*l*np.tan(angle)],
          '7': [1.5*l, 1.5*l*np.tan(angle)]
          }


# In[ ]:


for key, p in points.items():
    #print(key, p)
    plt.scatter(p[0], p[1], label=key)
#plt.legend()
plt.show()


# In[ ]:




