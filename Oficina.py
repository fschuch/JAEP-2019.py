#!/usr/bin/env python
# coding: utf-8

# <img src="JEAP.jpg" width="720">

# <h1>Sumário<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introdução" data-toc-modified-id="Introdução-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introdução</a></span><ul class="toc-item"><li><span><a href="#Sobre-o-autor" data-toc-modified-id="Sobre-o-autor-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Sobre o autor</a></span></li><li><span><a href="#Sobre-o-material" data-toc-modified-id="Sobre-o-material-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Sobre o material</a></span></li><li><span><a href="#Porque-Python?" data-toc-modified-id="Porque-Python?-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Porque Python?</a></span></li><li><span><a href="#Porque-Jupyter-Notebooks?" data-toc-modified-id="Porque-Jupyter-Notebooks?-1.4"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>Porque Jupyter Notebooks?</a></span></li><li><span><a href="#Material-Complementar" data-toc-modified-id="Material-Complementar-1.5"><span class="toc-item-num">1.5&nbsp;&nbsp;</span>Material Complementar</a></span></li></ul></li><li><span><a href="#Revisão" data-toc-modified-id="Revisão-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Revisão</a></span><ul class="toc-item"><li><span><a href="#Módulos" data-toc-modified-id="Módulos-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Módulos</a></span></li><li><span><a href="#Classes" data-toc-modified-id="Classes-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Classes</a></span></li><li><span><a href="#Dicionários" data-toc-modified-id="Dicionários-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Dicionários</a></span></li><li><span><a href="#Principais-Bibliotecas" data-toc-modified-id="Principais-Bibliotecas-2.4"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>Principais Bibliotecas</a></span></li><li><span><a href="#Boas-práticas-em-programação" data-toc-modified-id="Boas-práticas-em-programação-2.5"><span class="toc-item-num">2.5&nbsp;&nbsp;</span>Boas práticas em programação</a></span></li><li><span><a href="#Fortran-vs.-Python" data-toc-modified-id="Fortran-vs.-Python-2.6"><span class="toc-item-num">2.6&nbsp;&nbsp;</span>Fortran vs. Python</a></span></li></ul></li><li><span><a href="#Exercícios-Resolvidos" data-toc-modified-id="Exercícios-Resolvidos-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Exercícios Resolvidos</a></span><ul class="toc-item"><li><span><a href="#Métodos-numéricos" data-toc-modified-id="Métodos-numéricos-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Métodos numéricos</a></span></li><li><span><a href="#Fenômenos-de-Transporte" data-toc-modified-id="Fenômenos-de-Transporte-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Fenômenos de Transporte</a></span></li><li><span><a href="#Vibrações-Mecânicas" data-toc-modified-id="Vibrações-Mecânicas-3.3"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>Vibrações Mecânicas</a></span></li><li><span><a href="#Engenharia-Econômica" data-toc-modified-id="Engenharia-Econômica-3.4"><span class="toc-item-num">3.4&nbsp;&nbsp;</span>Engenharia Econômica</a></span></li><li><span><a href="#Eletronica" data-toc-modified-id="Eletronica-3.5"><span class="toc-item-num">3.5&nbsp;&nbsp;</span>Eletronica</a></span></li><li><span><a href="#Resistência-dos-Materiais" data-toc-modified-id="Resistência-dos-Materiais-3.6"><span class="toc-item-num">3.6&nbsp;&nbsp;</span>Resistência dos Materiais</a></span></li></ul></li><li><span><a href="#Exercícios-Propostos" data-toc-modified-id="Exercícios-Propostos-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Exercícios Propostos</a></span><ul class="toc-item"><li><span><a href="#Ciência-dos-Materiais" data-toc-modified-id="Ciência-dos-Materiais-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Ciência dos Materiais</a></span></li></ul></li></ul></div>

# ## Introdução
# 
# ### Sobre o autor
# 
# Possui graduação em Engenharia Mecânica pela Pontifícia Universidade Católica do Rio Grande do Sul (2013) e Mestrado em Engenharia e Tecnologia de Materiais pela mesma instituição. Atualmente atua como doutorando no Laboratório de Simulação de Escoamentos Turbulentos, Escola Politécnica da PUCRS. Possui experiencia em mecânica dos fluidos computacional, simulação numérica direta, simulação de grandes escalas, fenômenos de transporte, programação, programação paralela e métodos numéricos.
# 
# > **Felipe Nornberg Schuch**,<br>
# > Laboratório de Simulação de Escoamentos Turbulentos (LaSET),<br>
# > Escola Politécnica, Pontifícia Universidade Católica do Rio Grande do Sul.<br>
# > felipe.schuch@edu.pucrs.br

# ### Sobre o material

# ### Porque Python?
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

# * [10 motivos para você aprender Python](https://www.hostgator.com.br/blog/10-motivos-para-voce-aprender-python/)

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


# In[2]:


# Declarando variáveis
i = 5        #inteiro
f = 6.7      #ponto flutuante
g = 1e-2     #notação exponencial
s = 'abcdef' #string
c = 5.0 + 6j #complexo


# In[3]:


i = i + 1    #acumulador 
i


# In[4]:


i += 1       #forma alternativa para acumulador
i


# In[5]:


#laço de zero a 4
for i in range(5):
    print(i)


# In[6]:


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

# 1.  **SciPy**
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

# In[7]:


import scipy as sp
import scipy.sparse as sps


# Material complementar:
# * [SciPy](https://www.scipy.org/)
# * [Getting Started](https://www.scipy.org/getting-started.html)
# * [Scipy Lecture Notes](http://scipy-lectures.org/index.html)

# 2.  **Numpy**
# 
# ![Numpy](https://www.scipy.org/_static/images/numpylogo_med.png "Numpy")
# 
# Numpy é um pacote fundamental para a **computação científica em Python**. Entre outras coisas, destaca-se:
# * Objetos em arranjos N-dimensionais
# * Funções sofisticadas
# * Ferramentas para integrar código C/C++ e Fortran
# * Conveniente álgebra linear, transformada de Fourier e capacidade de números aleatórios
# 
# Além de seus usos científicos óbvios, o NumPy também pode ser usado como um contêiner multidimensional eficiente de dados genéricos. Tipos de dados arbitrários podem ser definidos. Isso permite que o NumPy integre-se de forma fácil e rápida a uma ampla variedade de bancos de dados.

# In[9]:


import numpy as np # Importando a biblioteca numpy e definindo-a com o codnome de np


# In[10]:


print(np.arange.__doc__) # É sempre possível checar a documentação de uma dada função


# In[11]:


a = np.arange(15).reshape(3, 5) # Criando um arranjo com 15 elementos e o redimensionando para o formato 3x5


# In[12]:


a # Escrevendo a


# In[13]:


a.shape # Verificando as dimensões do arranjo


# In[14]:


a.ndim # O número de dimensões


# In[15]:


a.dtype.name # Classificação quando ao tipo dos elementos


# In[16]:


a.itemsize # Tamanho em bytes de cada elemento


# In[17]:


a.size # Número total de elementos no arranjo


# In[18]:


type(a)


# In[19]:


# outras funções que merecem destaque:
for f in [np.zeros, np.zeros_like, np.ones, np.linspace]:
    print('=============== '+f.__name__+' ===============\n')
    print(f.__doc__+'\n')


# Material complementar:
# * [NumPy](https://www.numpy.org/)
# * [Quickstart tutorial](https://www.numpy.org/devdocs/user/quickstart.html)

# 3. **Pandas**
# 
# ![Pandas](https://www.scipy.org/_static/images/pandas_badge2.jpg "Pandas")
# 
# O pandas é um pacote Python que fornece **estruturas de dados rápidas, flexíveis e expressivas**, projetadas para tornar o trabalho com dados “relacionais” ou “rotulados” fáceis e intuitivos. O objetivo é ser o alicerce fundamental de alto nível para a análise prática de dados do mundo real em Python. Além disso, tem o objetivo mais amplo de se tornar a mais poderosa e flexível ferramenta de análise / manipulação de dados de código aberto disponível em qualquer linguagem.
# 
# Pandas é bem adequado para muitos tipos diferentes de dados:
# * Dados tabulares com colunas de tipos heterogêneos, como em uma tabela SQL ou planilha do Excel;
# * Dados de séries temporais ordenados e não ordenados (não necessariamente de frequência fixa);
# * Dados de matriz arbitrária (homogeneamente digitados ou heterogêneos) com rótulos de linha e coluna;
# * Qualquer outra forma de conjuntos de dados observacionais / estatísticos. Os dados realmente não precisam ser rotulados para serem colocados em uma estrutura de dados de pandas.

# In[20]:


import pandas as pd


# In[21]:


df2 = pd.DataFrame({'A': 1.,
                    'B': pd.Timestamp('20130102'),
                    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                    'D': np.array([3] * 4, dtype='int32'),
                    'E': pd.Categorical(["test", "train", "test", "train"]),
                    'F': 'foo'})


# In[22]:


df2


# In[23]:


print(pd.DataFrame.__doc__)


# In[24]:


#função bônus:
print(df2.to_latex())


# Material complementar:
# * [Pandas](https://pandas.pydata.org/)
# * [10 minutes to pandas](https://pandas.pydata.org/pandas-docs/version/0.25.0/getting_started/10min.html)

# 4. **Sympy**
# 
# ![Sympy](https://scipy.org/_static/images/sympy_logo.png "Sympy")
# 
# SymPy é uma biblioteca Python para **matemática simbólica**. O objetivo é tornar-se um sistema de álgebra computacional (CAS) completo, mantendo o código o mais simples possível para ser compreensível e facilmente extensível. SymPy é escrito inteiramente em Python.

# In[25]:


import sympy as sm
sm.init_printing() #Para escrever equações na tela


# In[26]:


x, t = sm.symbols('x t') # Criando símbolo


# calcular $\int (e^x \sin(x) + e^x \cos(x)) dx$

# In[27]:


sm.integrate(sm.exp(x)*sm.sin(x) + sm.exp(x)*sm.cos(x), x)


# calcular a derivada de $\sin(x)e^x$

# In[28]:


sm.diff(sm.sin(x)*sm.exp(x), x)


# calcular $\int_{-\infty}^{\infty} \sin(x^2)$
# 

# In[29]:


sm.integrate(sm.sin(x**2), (x, -sm.oo, sm.oo))


# calcular $\lim_{x \to 0} \dfrac{\sin(x)}{x}$

# In[30]:


sm.limit(sm.sin(x)/x, x, 0)


# resolver $x^2 - 2 = 0$

# In[31]:


sm.solve(x**2 - 2, x)


# resolver a equação diferencial $y'' - y = e^t$

# In[32]:


y = sm.Function('y')
eq1 = sm.dsolve(sm.Eq(y(t).diff(t, t) - y(t), sm.exp(t)), y(t))
eq1


# In[33]:


#Bônus
sm.latex(eq1)


# Material complementar:
# * [Sympy](https://www.sympy.org/en/index.html)
# * [Documentation](https://docs.sympy.org/latest/index.html)

# 5. **Matplotlib**
# 
# ![Matplotlib](https://www.scipy.org/_static/images/matplotlib_med.png "Matplotlib")
# 
# A Matplotlib é uma biblioteca de plotagem 2D do Python, que produz figuras de qualidade de publicação em uma variedade de formatos impressos e ambientes interativos entre plataformas. O Matplotlib pode ser usado em scripts Python, nos shells do Python e do IPython, no notebook Jupyter, nos servidores de aplicativos da web e em quatro kits de ferramentas de interface gráfica do usuário.
# 
# A **Matplotlib tenta tornar as coisas fáceis simples e as coisas difíceis possíveis**. Você pode gerar gráficos, histogramas, espectros de potência, gráficos de barras, gráficos de erros, diagramas de dispersão, etc., com apenas algumas linhas de código.
# 
# Material complementar:
# * [Matplotlib](https://matplotlib.org/)
# * [Style sheets reference](https://matplotlib.org/3.1.0/gallery/style_sheets/style_sheets_reference.html)
# * [Gallery](https://matplotlib.org/3.1.0/gallery/index.html)

# In[34]:


import matplotlib.pyplot as plt


# In[35]:


#Definindo um novo estilo para as figuras [opcional]
plt.style.use(['seaborn-notebook']) 


# In[36]:


x = np.arange(0, 5, 0.1)
y = np.sin(x)
plt.plot(x, y)


# 6. **Bokeh**
# 
#  ![Bokeh](https://bokeh.pydata.org/en/latest/_static/images/logo.png "Bokeh")
# 
# O Bokeh é uma biblioteca de visualização interativa para Python que permite uma apresentação visual de dados bonita e significativa em navegadores modernos. **Com o Bokeh, você pode criar, de maneira rápida e fácil, plotagens interativas, painéis e aplicativos de dados**.
# 
# O Bokeh fornece uma maneira elegante e concisa de construir gráficos versáteis e, ao mesmo tempo, oferecer interatividade de alto desempenho para conjuntos de dados grandes ou em fluxo.
# 
# Material complementar:
# * [Tutorial](https://mybinder.org/v2/gh/bokeh/bokeh-notebooks/master?filepath=tutorial%2F00%20-%20Introduction%20and%20Setup.ipynb)
# * [Gallery](https://bokeh.pydata.org/en/latest/docs/gallery.html)

# In[37]:


import bokeh as bk


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

# In[38]:


x = np.linspace(0., 2*np.pi, num=200, endpoint=True)

sinx = np.sin(x)
cosx = np.cos(x)

plt.plot(x, sinx, label='sin')
plt.plot(x, cosx, label='sin')

plt.legend()
plt.show()


# In[39]:


#Ou ainda:
for f in [np.sin, np.cos]:
    plt.plot(x, f(x), label=f.__name__)
    
plt.legend()
plt.show()


# In[40]:


del x, f


# ## Exercícios Resolvidos

# ### Métodos numéricos
# 
# 1. Diferenciação
# 
# 
# * Esquema em diferenças finitas, explícito, centrado e com precisão de O($\Delta x^2$):
# 
# \begin{equation}
#     \frac{\partial f}{\partial x} = f_i' = \dfrac{f_{i+1}-f_{i-1}}{2\Delta x}
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

# * Esquema em diferenças finitas, explícito, centrado e com precisão de O($\Delta x^2$):
# 
# \begin{equation}
#     \frac{\partial^2 f}{\partial x^2} = f_i'' = \dfrac{f_{i+1} - 2 f_{i} + f_{i-1}}{(\Delta x)^2}
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

# In[41]:


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


# Este é um bom momento para a criação de um módulo, já que operadores diferenciais serão muito utilizados durante os exemplos práticos. Para isso, vamos criar um arquivo nomeado `diferencial.py`, com as seguintes linhas de código:
# 
# ```python
# from scipy.sparse import diags
# from numpy import linspace, pi
# def ordem2(xf=2*pi, xi=0., nx=7, P=False):
#     '''ordem2(xf=2*pi, xi=0., nx=7, P=False)
# 
#     Calcula os operadores diferenciais da primeira e segunda derivada, para uma
#     malha equidistante de xi a xf, com nx pontos, e erro da ordem de dx**2.
# 
#     Parâmetros
#     ----------
#     xf : real
#          Valor do contorno superior de x.
#          Padrão é 2*pi.
#     xi : real
#          Valor do contorno inferior de x.
#          Padrão é zero.
#     nx : inteiro
#          Número de pontos da malha na direção x.
#          Padrão é 7.
#     P  : bool, opcional
#          Define que a condição de contorno é periódica quando True.
#          Padrão é False.
# 
#     Retorna
#     -------
#     x, Dx e Dx2, respectivamente o vetor posição e os operadores diferenciais
#     para primeira e segunda ordem.
#     '''
#     x = linspace(xi, xf, num=nx, endpoint=not P)
#     dx = (x[1]-x[0])
#     #Operador diferencial de primeira ordem
#     Dx = diags([-1., 1.],
#               offsets=[-1, 1],
#               shape=(x.size,x.size)
#               ).toarray()
#     #Operador diferencial de segunda ordem
#     Dx2 = diags([1., -2., 1.],
#                offsets=[-1, 0, 1],
#                shape=(x.size,x.size)
#                ).toarray()
#     #
#     if P: #Condições de contorno Periódicas
#         for D in [Dx, Dx2]:
#             D[0,-1] = D[1,0]
#             D[-1,0] = D[-2,-1]
#     else: #Não Periódica
#         Dx[0,0], Dx[0,1], Dx[0,2] = -3., 4., -1.
#         Dx[-1,-3], Dx[0,-2], Dx[0,-1] = 1., -4., 3.
#         Dx2[0,0], Dx2[0,1], Dx2[0,2] = 1., -2., 1.
#         Dx[-1,-3], Dx[0,-2], Dx[0,-1] = 1., -2., 1.
#     #
#     Dx /= 2*dx
#     Dx2 /= dx**2.
#     return x, Dx, Dx2
# ```
# 
# Agora pode-se importar o novo módulo com:

# In[42]:


import diferencial as dv

x, Dx, Dx2 = dv.ordem2(n=100, P=False)

f = np.cos

plt.plot(x, f(x), label='f(x)')
plt.plot(x, Dx.dot(f(x)), label="f'(x)")
plt.plot(x, Dx2.dot(f(x)), label="f''(x)")

plt.legend()
plt.show()


# * Esquema em diferenças finitas, explícito, diferença para frente e com precisão de O($\Delta t^1$):
# 
# \begin{equation}
#     \frac{\partial f}{\partial t} = \dfrac{f_{k+1}-f_{k}}{\Delta t}
# \end{equation}

# * Esquema em diferenças finitas, implícito, centrado e com precisão de O($\Delta x^6$):
# 
# \begin{equation}
#     \frac{1}{3} f_{i-1}' + f_{i}' + \frac{1}{3} f_{i+1}' = \frac{14}{9} \frac{f_{i+1}-f_{i-1}}{2\Delta x} + \frac{1}{9} \frac{f_{i+2}-f_{i-2}}{4\Delta x}
# \end{equation}
# 
# \begin{equation}
# \begin{split}
# \underbrace{
# \begin{bmatrix}
# 1 & 1/3 & & & & & 1/3 \\
# 1/3 & 1 & 1/3 & & \\
#  & \ddots & \ddots & \ddots & & & \\
# & & 1/3 & 1 & 1/3 & \\
# & & & \ddots & \ddots & \ddots & \\
# & & & & 1/3 & 1 & 1/3\\
# 1/3 & & & & & 1/3 & 1
# \end{bmatrix}
# }_{A}
# \begin{bmatrix} f'_{0} \\ f'_{1} \\ \vdots \\ f'_{i} \\ \vdots \\ f'_{n-2}\\ f'_{n-1}\end{bmatrix} =
# \underbrace{
# \frac{1}{\Delta x}
# \begin{bmatrix}
# 0 & 7/9 & 1/36 & & & 1/36 & 7/9 \\
# 7/9 & 0 & 7/9 & 1/36 & & & 1/36\\
# & \ddots & \ddots & \ddots & & & \\
# & 1/36 & 7/9 & 0 & 7/9 & 1/36 \\
# & & & \ddots & \ddots & \ddots & \\
# 1/36 & & & 1/36 & 7/9 & 0 & 7/9 \\
# 7/9 & 1/36 & & & 1/36 & 7/9 & 0
# \end{bmatrix}
# }_{B}
# \begin{bmatrix} f_{0} \\ f_{1} \\ \vdots \\ f_{i} \\ \vdots \\ f_{n-2}\\ f_{n-1}\end{bmatrix}
# \\
# \mbox{com $2\leq i \leq n -3$, $f_0 = f_n$ e $f_0' = f_n'$}
# \end{split}
# \label{eq.dxx_matrix}
# \end{equation}
# 
# \begin{equation}
#     f' = \underbrace{A^{-1}B}_{D_x} f
# \end{equation}

# 2. Integração
# 
# 
# * Regra do Trapézio:
# 
# \begin{equation}
# \int_a^b f(x)dx \approx \dfrac{f(a) + f(b)}{2} (b-a)
# \end{equation}
# 
# Pode-se dividir em $N$ intervalos
# 
# \begin{equation}
# \int_a^b fdx \approx \sum_{i=0}^{N-1} \dfrac{f_{i} + f_{i+1}}{2} \Delta x = \dfrac{\Delta x}{2} \left( f_0 + 2f_1 + \dots + 2f_{n-1} + f_{n}\right)
# \end{equation}
# 
# Escrito na forma matricial:
# 
# \begin{equation}
# \int_a^b fdx = \sum \left(
# \underbrace{
# \Delta x
# \begin{bmatrix}
# 1/2 & & & & & & \\
# & 1 & & & & & \\
# & & \ddots & & & & \\
# & & & 1 & & \\
# & & & & \ddots & \\
# & & & & & 1 & \\
# & & & & & & 1/2
# \end{bmatrix}
# }_{I_x = \text{ Operador integral}}
# \begin{bmatrix} f_{0} \\ f_{1} \\ \vdots \\ f_{i} \\ \vdots \\ f_{n-1}\\ f_{n}\end{bmatrix}
# \right)
# \end{equation}

# In[43]:


x = np.linspace(0., 2*np.pi, num=201, endpoint=True)

f, dx = np.sin(x), (x[1]-x[0])

Ix = dx*sps.eye(x.size).toarray() #Operador integral

for i in [0, -1]: #Condições de contorno
    Ix[i] *= 0.5

Ix.dot(f).sum() #Integral


# In[44]:


#Ou alternativamente, utilizando o scipy:
import scipy.integrate
sp.integrate.trapz(f,x)


# Material complementar:
# * [Finite Difference Coefficients Calculator](http://web.media.mit.edu/~crtaylor/calculator.html)
# * [Trapezoidal rule](https://en.wikipedia.org/wiki/Trapezoidal_rule)
# * MOIN, Parviz. **Fundamentals of engineering numerical analysis**. Cambridge University Press, 2010.

# ### Fenômenos de Transporte
# 
# * Transferência de Calor, massa e quantidade de movimento

# 1. **Radiação e convecção combinadas em transferência de calor permanente unidirecional**. A superfície interior de uma parede de espessura $L=0,25m$ é mantida a $21^oC$, enquanto a temperatura no meio externo é $T_\infty = 4^oC$. Considere que há troca de calor com a vizinhança $T_{viz} = 250 K$, o coeficiente de convecção é $h=23W/m^2$, a condutividade térmica do material que compõe a parede é $k=0,65 W/m \cdot ^oC$, a emissividade da superfície externa vale $\epsilon = 0,8$ e a constante de Stefan-Boltzmann $\sigma = 5,67 \times 10^{-8} [W/m^2 \cdot K^4]$. Determine a temperatura externa da parede $T_2$.
# 
# \begin{equation}
#     k \left( \dfrac{T_1-T_2}{L} \right) = \epsilon \sigma \left( T_2^4 - T_{viz}^4 \right) + h \left( T_2 - T_\infty \right)
# \end{equation}
# 
# 

# In[45]:


k, t1, t2, l, epsi, sigma, tviz, too, h = sm.symbols("k t_1 t_2 L \epsilon \sigma t_{viz} t_\infty h")

eq1 = sm.Eq(k*(t1-t2)/l,epsi*sigma*(t2**4.-tviz**4.)+h*(t2-too))

#Dicionário com os valores que serão substituidos na eq1
dic = {k: .65, t1: 21.+273., l: .25, epsi: .8, sigma: 5.67e-8, tviz: 255., too: 4.+273., h: 23.}

sol = sm.solve(eq1.subs(dic), t2)

for val in sol:
    if val.is_real: #A solução deve ser real
        if val.is_positive: #E positiva
            print('T2 = {0:3.2f} graus celsius'.format(val-273.))  


# In[46]:


del k, t1, t2, l, epsi, sigma, tviz, too, h, eq1, dic, sol


# 2. **Condição de calor transiente bidimensional**. Uma placa de cobre de $50cm \times 50cm$ inicialmente possui temperatura em toda a sua extensão igual a $0^oC$. Instantaneamente , suas bordas são levadas às temperaturas de $60^oC$ em $x=0$; $20^oC$ em $x=50cm$; $0^oC$ em $y=0$ e $100^oC$ em $y=50$. A difusividade térmica do cobre é $1,1532cm^2/s$. Considerando um $\Delta t = 4s$, $\Delta x = \Delta y = 5cm$, calcule a evolução da temperatura para a posição central da placa até o tempo de $400s$. Para o tempo de $200s$ apresente o perfil de temperatura em todos os pontos discretos do domínio.
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

# In[47]:


x = np.linspace(0., 50., num=11, endpoint=True)
y = np.linspace(0., 50., num=11, endpoint=True)
t = np.linspace(0., 400., num=101, endpoint=True)

a = 1.1532

T = np.zeros((x.size,y.size,t.size))

T[0,:,:], T[-1,:,:], T[:,0,:], T[:,-1,:] = 60., 20., 0., 100.


# In[48]:


dt = t[1]-t[0]
dx2 = (x[1]-x[0])**2.
dy2 = (y[1]-y[0])**2.
for n in range(t.size-1):
    for i in range(1,x.size-1):
        for j in range(1,y.size-1):
            T[i,j,n+1] = dt*a*((T[i-1,j,n]-2.*T[i,j,n]+T[i+1,j,n])/dx2+(T[i,j-1,n]-2.*T[i,j,n]+T[i,j+1,n])/dy2)+T[i,j,n]


# In[49]:


#Adicionar subplot
fig, ax = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True)

for i, n in enumerate(T[:,:,::25].T):
    ax[i].imshow(n)
    ax[i].set_title('t={}'.format(t[i*25]))
    
plt.show()


# In[50]:


plt.plot(t, T[5,5,:])
plt.title('Evolução da temperatura no centro da placa')
plt.xlabel('t')
plt.ylabel('T')
plt.show()


# In[51]:


del T, x, y, t, a, dx2, dy2, dt, i, n


# 3. **Condução e difusão transiente unidimensional**. Resolver a EDP:
# \begin{equation}
#     \dfrac{\partial T}{\partial t}+u\dfrac{\partial T}{\partial x}=\alpha \dfrac{\partial^2 T}{\partial x^2} \quad 0\leq x \leq 1 ; 0\leq t \leq 8
# \end{equation}
# Condições de contorno:
# \begin{equation}
#     T(0,t)=T(1,t)=0
# \end{equation}
# Condição inicial:
# \begin{equation}
#     T(x,0) =  1 - ( 10 x - 1 )^2 \quad \text{ se $0 \leq x \leq 0,2$}, \quad \text{ senão } T(x,0) = 0
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

# In[165]:


x, Dx, Dx2 = dv.ordem2(1., n=101)

t = np.linspace(0., 8., num=8001, endpoint=True)

dt = t[1]-t[0]

#Condições de contorno
for D in [Dx, Dx2]:
    D[0,:] = 0.
    D[-1,:] = 0.


# In[166]:


def convdiff(alpha, u):

    #Condição inicial
    T = np.zeros((x.size))
    for i, ival in enumerate(x):
        if ival > 0.2:
            break
        T[i] = 1. - (10. * ival - 1)**2.

    A = dt*(alpha*Dx2-u*Dx)
    
    return A, T


# In[167]:


alpha, u, visu = 0.001, 0.08, 2000 #Parâmetro de visualização: a cada quantos passos de tempo se deve graficar os resultados

A, T = convdiff(alpha, u)
for n in range(t.size):
    T += A.dot(T)
    if n % visu == 0:
        plt.plot(x, T, label='t={}'.format(t[n]))

plt.xlabel('x')
plt.ylabel('T(x)')
plt.legend()
plt.show()


# Mas o que acontece quando combinados diferentes valores para $\alpha$ e $u$?

# In[168]:


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


# In[169]:


del x, t, dt, D, Dx, Dx2, visu, alpha, u, A, T
plt.close('all')


# ### Vibrações Mecânicas
# 
# 
# 
# \begin{equation}
# \underbrace{
# \begin{bmatrix}
# m_1 & 0 \\
#   0 & m_2
# \end{bmatrix}
# }_{M}
# \begin{bmatrix} \ddot x_0 \\ \ddot x_1 \end{bmatrix} +
# \underbrace{
# \begin{bmatrix}
# c_1 + c_2 & -c_2 \\
# -c_2 & c_2 + c_3
# \end{bmatrix}
# }_{C}
# \begin{bmatrix} \dot x_0 \\ \dot x_1 \end{bmatrix} +
# \underbrace{
# \begin{bmatrix}
# k_1 + k_2 & -k_2 \\
# -k_2 & k_2 + k_3
# \end{bmatrix}
# }_{K}
# \begin{bmatrix} x_0 \\ x_1 \end{bmatrix} = 0
# \end{equation}
# 
# \begin{equation}
#     x = a e^{iwt}
# \end{equation}
# 
# \begin{equation}
#     \left[ \underbrace{M^{-1}K}_{A} - w^2I \right] x = 0
# \end{equation}

# In[150]:


m1, m2 = 1., 1.
k1, k2, k3 = 1., 1., 1.

M = np.array([[m1, 0.],
              [0., m2]])
K = np.array([[k1+k2, -k2  ],
              [-k2  , k2+k3]])

from scipy import linalg
A = sp.linalg.inv(M).dot(K)

e_vals, e_vecs = LA.eig(A)


# In[151]:


fig, ax = plt.subplots(nrows=1, ncols=e_vals.size, sharex=True, sharey=True)

t = np.linspace(0., 4*np.pi, num=200, endpoint=True)

for i, w in enumerate(e_vals):
    for j, a in enumerate(e_vecs[i]):
        offset = -(-1)**j #um deslocamento, apenas para melhor visualização
        ax[i].plot(offset+a*np.exp(1j*w*t),t, label=r'$x_{}$'.format(j))
    ax[i].legend()
    ax[i].set_title(r'$\omega_{} = {}$'.format(i,w.real))
    ax[i].set_xlabel('x')
    ax[i].set_ylabel('t')
plt.show()


# ### Engenharia Econômica

# ### Eletronica
# 
# 
# 1. Determine as correntes elétricas $I_1$, $I_2$ e $I_3$ que percorrem o circuito abaixo, admitindo que: $R_1=10 \Omega$, $R_2=22\Omega$, $R_3=47\Omega$, $R_4=22\Omega$, $R_5=10\Omega$ e $V_1=V_2=9 V$.
# 
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

# In[57]:


r1, r2, r3, r4, r5 = 10., 22., 47., 22., 10.
v1, v2 = 9., 9.

A = [[1., 1., -1.],
     [r1+r4, 0., r3],
     [0., r2+r5, r3]]

B = [0., v1, v2]


# In[58]:


I = np.linalg.solve(A,B)

for i, ival in enumerate(I):
    print('I{} = {}'.format(i, ival))


# ### Resistência dos Materiais
# 
# 1. Determine os esforços am cada uma das barras que compôem a treliça abaixo. Para cada um dos nós observa-se que que $\sum F_x = 0$ e $\sum F_y = 0$ dem ser satisfeitas, resultando em 14 equações, e um sistema matricial $14 \times 14$ a ser resolvido. Como incógnitas, tem-se a força resultante nas 11 barras $F_{ij}$, além da força nos apoios: $F_{1x}$, $F_{1y}$ e $F_{4y}$. Considera o carregamento como $F_5 = F_6 = 1kN$ e $F_7 = 2kN$.
# 
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

# In[59]:


angle = np.deg2rad(30.)
F5, F6, F7 = 1., 1., 2.
cos, sin = np.cos(angle), np.sin(angle)

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


# In[73]:


I = np.linalg.solve(A,B)

for i, key in enumerate(['F1x','F1y','F4y','F15','F12','F23','F25','F27','F34','F36','F37','F46','F57','F67']):
    print('{0:2d}: {1} = {2:5.2f}kN'.format(i,key,I[i]))


# ## Exercícios Propostos

# ### Ciência dos Materiais
# 
# 1. **Difusão combinada de massa e calor em um caso bidimensional**.
# 
# \begin{equation}
# \alpha \left( \dfrac{\partial ^2 T}{\partial x^2} + \dfrac{\partial ^2 T}{\partial y^2} \right) =\dfrac{\partial T}{\partial t}
# \end{equation}
# 
# \begin{equation}
# \alpha \left( \dfrac{\partial ^2 T}{\partial x^2} + \dfrac{\partial ^2 T}{\partial y^2} \right) =\dfrac{\partial T}{\partial t}
# \end{equation}

# In[180]:


prop = pd.DataFrame({'Soluto': pd.Categorical(["Carbono", "Carbono", "Ferro", "Ferro", "Níquel", "Manganês", "Zinco", "Cobre", "Cobre", "Prata", "Prata", "Carbono"]),
                     'Solvente': pd.Categorical(["Ferro CFC", "Ferro CCC", "Ferro CFC", "Ferro CCC", "Ferro CFC", "Ferro CFC", "Cobre", "Alumínio", "Cobre", "Prata (Cristal)", "Prata (Cont.Grão)", "Titânio"]),
                     'alpha-500': [5e-15, 1e-12, 2e-23, 1e-20, 1e-23, 3e-24, 4e-18, 4e-14, 1e-18, 1e-17, 1e-11, 3e-16],
                     'alpha-1000': [3e-11, 2e-9, 2e-16, 3e-14, 2e-16, 1e-16, 5e-13, 1e-10, 2e-11, 1e-12, np.NaN, 2e-11]
                    })
prop


# In[ ]:




