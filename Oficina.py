#!/usr/bin/env python
# coding: utf-8

# <img src="JEAP.jpg" width="480">

# <h1>Sumário<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introdução" data-toc-modified-id="Introdução-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introdução</a></span><ul class="toc-item"><li><span><a href="#Sobre-o-autor" data-toc-modified-id="Sobre-o-autor-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Sobre o autor</a></span></li><li><span><a href="#Sobre-o-material" data-toc-modified-id="Sobre-o-material-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Sobre o material</a></span></li><li><span><a href="#Porque-Python?-ref" data-toc-modified-id="Porque-Python?-ref-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Porque Python? <a href="https://www.hostgator.com.br/blog/10-motivos-para-voce-aprender-python/" target="_blank">ref</a></a></span></li><li><span><a href="#Porque-Jupyter-Notebooks?" data-toc-modified-id="Porque-Jupyter-Notebooks?-1.4"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>Porque Jupyter Notebooks?</a></span></li><li><span><a href="#Material-Complementar" data-toc-modified-id="Material-Complementar-1.5"><span class="toc-item-num">1.5&nbsp;&nbsp;</span>Material Complementar</a></span></li></ul></li><li><span><a href="#Revisão" data-toc-modified-id="Revisão-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Revisão</a></span><ul class="toc-item"><li><span><a href="#Módulos" data-toc-modified-id="Módulos-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Módulos</a></span></li><li><span><a href="#Classes" data-toc-modified-id="Classes-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Classes</a></span></li><li><span><a href="#Dicionários" data-toc-modified-id="Dicionários-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Dicionários</a></span></li><li><span><a href="#Principais-Bibliotecas" data-toc-modified-id="Principais-Bibliotecas-2.4"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>Principais Bibliotecas</a></span><ul class="toc-item"><li><span><a href="#SciPy" data-toc-modified-id="SciPy-2.4.1"><span class="toc-item-num">2.4.1&nbsp;&nbsp;</span>SciPy</a></span></li><li><span><a href="#Numpy" data-toc-modified-id="Numpy-2.4.2"><span class="toc-item-num">2.4.2&nbsp;&nbsp;</span>Numpy</a></span></li><li><span><a href="#Exemplos" data-toc-modified-id="Exemplos-2.4.3"><span class="toc-item-num">2.4.3&nbsp;&nbsp;</span>Exemplos</a></span></li><li><span><a href="#Pandas" data-toc-modified-id="Pandas-2.4.4"><span class="toc-item-num">2.4.4&nbsp;&nbsp;</span>Pandas</a></span></li><li><span><a href="#Matplotlib" data-toc-modified-id="Matplotlib-2.4.5"><span class="toc-item-num">2.4.5&nbsp;&nbsp;</span>Matplotlib</a></span></li><li><span><a href="#Bokeh" data-toc-modified-id="Bokeh-2.4.6"><span class="toc-item-num">2.4.6&nbsp;&nbsp;</span>Bokeh</a></span></li></ul></li><li><span><a href="#Boas-práticas-em-programação" data-toc-modified-id="Boas-práticas-em-programação-2.5"><span class="toc-item-num">2.5&nbsp;&nbsp;</span>Boas práticas em programação</a></span></li><li><span><a href="#Fortran-vs.-Python" data-toc-modified-id="Fortran-vs.-Python-2.6"><span class="toc-item-num">2.6&nbsp;&nbsp;</span>Fortran vs. Python</a></span></li></ul></li><li><span><a href="#Exercícios-Resolvidos" data-toc-modified-id="Exercícios-Resolvidos-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Exercícios Resolvidos</a></span><ul class="toc-item"><li><span><a href="#Métodos-Numéricos" data-toc-modified-id="Métodos-Numéricos-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Métodos Numéricos</a></span></li><li><span><a href="#Transferência-de-Calor" data-toc-modified-id="Transferência-de-Calor-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Transferência de Calor</a></span><ul class="toc-item"><li><span><a href="#1D-Radiação-+-Convecção" data-toc-modified-id="1D-Radiação-+-Convecção-3.2.1"><span class="toc-item-num">3.2.1&nbsp;&nbsp;</span>1D Radiação + Convecção</a></span></li></ul></li><li><span><a href="#Mecânica-dos-Fluidos" data-toc-modified-id="Mecânica-dos-Fluidos-3.3"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>Mecânica dos Fluidos</a></span></li><li><span><a href="#Vibrações-Mecânicas" data-toc-modified-id="Vibrações-Mecânicas-3.4"><span class="toc-item-num">3.4&nbsp;&nbsp;</span>Vibrações Mecânicas</a></span></li><li><span><a href="#Resistência-dos-Materiais" data-toc-modified-id="Resistência-dos-Materiais-3.5"><span class="toc-item-num">3.5&nbsp;&nbsp;</span>Resistência dos Materiais</a></span></li><li><span><a href="#Ciência-dos-Materiais" data-toc-modified-id="Ciência-dos-Materiais-3.6"><span class="toc-item-num">3.6&nbsp;&nbsp;</span>Ciência dos Materiais</a></span></li><li><span><a href="#Controle-e-Automação" data-toc-modified-id="Controle-e-Automação-3.7"><span class="toc-item-num">3.7&nbsp;&nbsp;</span>Controle e Automação</a></span></li><li><span><a href="#Engenharia-Econômica" data-toc-modified-id="Engenharia-Econômica-3.8"><span class="toc-item-num">3.8&nbsp;&nbsp;</span>Engenharia Econômica</a></span></li></ul></li><li><span><a href="#Exercícios-Propostos" data-toc-modified-id="Exercícios-Propostos-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Exercícios Propostos</a></span></li></ul></div>

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

# In[81]:


x = np.linspace(0., 2*np.pi, num=200, endpoint=True)

for f in [np.sin, np.cos]:
    plt.plot(x, f(x), label=f.__name__)
    
plt.legend()
plt.show()


# In[82]:


del x, f


# ## Exercícios Resolvidos

# In[6]:


import numpy as np
import scipy as sp
import sympy as sm
import bokeh as bk
import pandas as pd
import matplotlib.pyplot as plt

sm.init_printing()


# ### Métodos Numéricos

# ### Transferência de Calor

# #### 1D Radiação + Convecção

# In[33]:


k, t1, t2, l, epsi, sigma, tviz, too, h = sm.symbols("k t_1 t_2 L \epsilon \sigma t_{viz} t_\infty h")


# In[34]:


eq1 = sm.Eq(k*(t1-t2)/l,epsi*sigma*(t2**4.-tviz**4.)+h*(t2-too))
eq1


# In[39]:


print(eq1.subs.__doc__)


# In[42]:


dic = {k: 1., t1: 0., l: 1., epsi: 1., sigma: 1., tviz: 1., too: 1., h: 1.}
eq1.subs(dic)


# In[44]:


print(sm.solve.__doc__)


# In[55]:


sol = sm.solve(eq1.subs(dic), t2)


# In[69]:


for val in sol:
    if val.is_real:
        if val.is_positive:
            print(val)  


# In[32]:


del k, t1, t2, l, epsi, sigma, tviz, too, h, eq1, dic, sol


# ### Mecânica dos Fluidos

# ### Vibrações Mecânicas

# ### Resistência dos Materiais

# ### Ciência dos Materiais

# ### Controle e Automação

# ### Engenharia Econômica

# ## Exercícios Propostos

# In[ ]:




