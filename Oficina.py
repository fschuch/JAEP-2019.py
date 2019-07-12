#!/usr/bin/env python
# coding: utf-8

# <img src="JEAP.jpg" width="720">

# <h1>Sumário<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introdução" data-toc-modified-id="Introdução-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introdução</a></span><ul class="toc-item"><li><span><a href="#Sobre-o-autor" data-toc-modified-id="Sobre-o-autor-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Sobre o autor</a></span></li><li><span><a href="#Sobre-o-material" data-toc-modified-id="Sobre-o-material-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Sobre o material</a></span></li><li><span><a href="#Porque-Python?" data-toc-modified-id="Porque-Python?-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Porque Python?</a></span></li><li><span><a href="#Porque-Jupyter-Notebooks?" data-toc-modified-id="Porque-Jupyter-Notebooks?-1.4"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>Porque Jupyter Notebooks?</a></span></li><li><span><a href="#Material-Complementar" data-toc-modified-id="Material-Complementar-1.5"><span class="toc-item-num">1.5&nbsp;&nbsp;</span>Material Complementar</a></span></li></ul></li><li><span><a href="#Revisão" data-toc-modified-id="Revisão-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Revisão</a></span><ul class="toc-item"><li><span><a href="#Listas" data-toc-modified-id="Listas-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Listas</a></span></li><li><span><a href="#Dicionários" data-toc-modified-id="Dicionários-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Dicionários</a></span></li><li><span><a href="#Módulos" data-toc-modified-id="Módulos-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Módulos</a></span></li><li><span><a href="#Principais-Bibliotecas" data-toc-modified-id="Principais-Bibliotecas-2.4"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>Principais Bibliotecas</a></span></li><li><span><a href="#Fortran-vs.-Python" data-toc-modified-id="Fortran-vs.-Python-2.5"><span class="toc-item-num">2.5&nbsp;&nbsp;</span>Fortran vs. Python</a></span></li></ul></li><li><span><a href="#Exercícios-Resolvidos" data-toc-modified-id="Exercícios-Resolvidos-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Exercícios Resolvidos</a></span><ul class="toc-item"><li><span><a href="#Métodos-numéricos" data-toc-modified-id="Métodos-numéricos-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Métodos numéricos</a></span></li><li><span><a href="#Fenômenos-de-Transporte" data-toc-modified-id="Fenômenos-de-Transporte-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Fenômenos de Transporte</a></span></li><li><span><a href="#Vibrações-Mecânicas" data-toc-modified-id="Vibrações-Mecânicas-3.3"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>Vibrações Mecânicas</a></span></li><li><span><a href="#Engenharia-Econômica" data-toc-modified-id="Engenharia-Econômica-3.4"><span class="toc-item-num">3.4&nbsp;&nbsp;</span>Engenharia Econômica</a></span></li><li><span><a href="#Eletrônica" data-toc-modified-id="Eletrônica-3.5"><span class="toc-item-num">3.5&nbsp;&nbsp;</span>Eletrônica</a></span></li><li><span><a href="#Resistência-dos-Materiais" data-toc-modified-id="Resistência-dos-Materiais-3.6"><span class="toc-item-num">3.6&nbsp;&nbsp;</span>Resistência dos Materiais</a></span></li></ul></li><li><span><a href="#Exercícios-Propostos" data-toc-modified-id="Exercícios-Propostos-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Exercícios Propostos</a></span></li></ul></div>

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
# 
# * O objetivo desta palestra é **introduzir os principais conceitos empregados em programação e Python**, mais especificamente no contexto interativo da plataforma Jupyter Notebook;
# * Além de demonstrar como **solucionar diversos problemas de engenharia** por meio de propostas computacionais;
# * Para tanto, o material inclui uma breve **revisão de conceitos fundamentais**, estruturas de dados e as principais bibliotecas científicas disponíveis. Para maiores detalhes, **pode-se consultar a documentação disponível** ou mesmo as diversas leituras recomendadas que aparecem no decorrer do texto.
# * Finalmente, **a prática leva a perfeição**, há uma série de exercícios propostos ao final do material, visando a fixação do conhecimento.

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
# * [Why I write with LaTeX (and why you should too)](https://medium.com/@marko_kovic/why-i-write-with-latex-and-why-you-should-too-ba6a764fadf9)
# * [New Developer? You should’ve learned Git yesterday](https://codeburst.io/number-one-piece-of-advice-for-new-developers-ddd08abc8bfa)

# ## Revisão

# In[ ]:


'''
Isso é um comentário
'''

print("Olá mundo")

# Isso também é um comentário


# In[ ]:


# Declarando variáveis
i = 5        #inteiro
f = 6.7      #ponto flutuante
g = 1e-2     #notação exponencial
s = 'abcdef' #string
c = 5.0 + 6j #complexo


# In[ ]:


i = i + 1    #acumulador 
i


# In[ ]:


i += 1       #forma alternativa para acumulador
i


# In[ ]:


#laço de zero a 4
for i in range(5):
    print(i)


# In[ ]:


#teste lógico
if i == 4:
    print('i é igual a 4')
else:
    print('i não é igual a 4, i é igual a '+str(i))


# Material complementar:
# 
# * [More Control Flow Tools](https://docs.python.org/3/tutorial/controlflow.html)

# ### Listas
# 
# * Um exemplo dos principais métodos:

# In[ ]:


fruits = ['orange', 'apple', 'pear', 'banana', 'kiwi', 'apple', 'banana']


# In[ ]:


fruits[2]


# In[ ]:


fruits.count('apple')


# In[ ]:


fruits.count('tangerine')


# In[ ]:


fruits.index('banana')


# In[ ]:


fruits.index('banana', 4)  # Find next banana starting a position 4


# In[ ]:


fruits.reverse()
fruits


# In[ ]:


fruits.append('grape')
fruits


# In[ ]:


fruits.sort()
fruits


# In[ ]:


fruits.pop()


# In[ ]:


for f in fruits:
    print(f)


# In[ ]:


for i, f in enumerate(fruits):
    print(i,f)


# * Usando listas como pilha:
# Os métodos de lista facilitam muito o uso de uma lista como uma pilha (*Stacks*), onde o último elemento adicionado é o primeiro elemento recuperado (*“last-in, first-out”*). Para adicionar um item ao topo da pilha, use `append()`. Para recuperar um item do topo da pilha, use `pop()` sem um índice explícito. Por exemplo:

# In[ ]:


stack = [3, 4, 5]
stack.append(6)
stack.append(7)
stack


# In[ ]:


stack.pop()


# In[ ]:


stack.pop()


# In[ ]:


stack.pop()


# In[ ]:


stack


# ### Dicionários
# 
# Ao contrário das seqüências, que são indexadas por um intervalo de números, **os dicionários são indexados por chaves (*keys*)**, que podem ser de qualquer tipo imutável; *strings* e números sempre podem ser usados como chave.
# 
# É melhor pensar em um dicionário como um conjunto de pares `key: value`, com a exigência de que as chaves sejam **exclusivas** (dentro de um dicionário).
# 
# As operações principais em um dicionário estão armazenando um valor com alguma chave e extraindo o valor dado a chave. Também é possível excluir um par `key: value` com `del`. **Se você armazenar usando uma chave que já está em uso, o valor antigo associado a essa chave será esquecido**. É um erro extrair um valor usando uma chave inexistente.
# 
# A execução de `list()` em um dicionário retorna uma lista de todas as chaves usadas no dicionário, em ordem de inserção (se você quiser classificá-las, use apenas `sorted()`). Para verificar se uma única chave está no dicionário, use a palavra-chave `in`.

# In[ ]:


tel = {'jack': 4098, 'sape': 4139}
tel['guido'] = 4127
tel


# In[ ]:


tel['jack']


# In[ ]:


del tel['sape']
tel['irv'] = 4127
tel


# In[ ]:


list(tel)


# In[ ]:


sorted(tel)


# In[ ]:


'guido' in tel


# In[ ]:


'jack' not in tel


# In[ ]:


for key, t in tel.items():
    print(key, t)


# ### Módulos
# 
# Se você sair do interpretador Python e voltar novamente, as definições feitas (funções e variáveis) serão perdidas. Portanto, se você quiser escrever um programa um pouco mais longo, é melhor usar um editor de texto para preparar a entrada para o interpretador e executá-lo com esse arquivo como entrada. **Isso é conhecido como criar um script**. À medida que seu programa fica mais longo, você pode querer dividi-lo em vários arquivos para **facilitar a manutenção**. Você também pode usar uma função útil que tenha escrito em **vários programas sem copiar sua definição** em cada programa.
# 
# Para suportar isso, o Python tem uma maneira de colocar as definições em um arquivo e usá-las em um script ou em uma instância interativa do interpretador. **Esse arquivo é chamado de módulo**; As definições de um módulo podem ser importadas para outros módulos ou para o módulo principal.
# 
# Um módulo é um arquivo contendo definições e instruções do Python. O nome do arquivo é o nome do módulo com o sufixo `.py` acrescentado. Dentro de um módulo, o nome do módulo (como uma string) está disponível como o valor da variável global `__name__`.

# Por exemplo, use seu editor de texto favorito para criar um arquivo chamado `fibo.py` no diretório atual com o seguinte conteúdo:
# 
# ```Python
# # Fibonacci numbers module
# 
# def fib(n):    # write Fibonacci series up to n
#     a, b = 0, 1
#     while a < n:
#         print(a, end=' ')
#         a, b = b, a+b
#     print()
# 
# def fib2(n):   # return Fibonacci series up to n
#     result = []
#     a, b = 0, 1
#     while a < n:
#         result.append(a)
#         a, b = b, a+b
#     return result
# ```

# In[ ]:


import fibo


# In[ ]:


fibo.fib(1000)


# In[ ]:


fibo.fib2(100)


# In[ ]:


fibo.__name__


# In[ ]:


dir(fibo)


# Material complementar:
# 
# * [Python - Modules](https://www.tutorialspoint.com/python/python_modules)
# * [The Python Tutorial - Modules](https://docs.python.org/3/tutorial/modules.html)
# * [Data Structures](https://docs.python.org/3/tutorial/datastructures.html)
# * [Dictionaries in Python](https://realpython.com/python-dicts/)
# * [Classes](https://docs.python.org/2/tutorial/classes.html)

# In[ ]:


del c, f, fruits, g, i, key, s, stack, t, tel


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
#     * Numpy
#     * Matplotlib
#     * Sympy
#     * IPython
#     * Pandas

# * Além disso, a própria biblioteca SciPy, um componente do conjunto SciPy, fornecendo muitas rotinas numéricas:
#     * Funções especiais
#     * Integração numérica
#     * Diferenciação numérica
#     * Otimização
#     * Interpolação
#     * Transformada de Fourier
#     * Processamento de sinal
#     * Algebra linear e Algebra linear esparsa
#     * Problema de autovalor esparso com ARPACK
#     * Algoritmos e estruturas de dados espaciais
#     * Estatistica
#     * Processamento de imagem multidimensional
#     * I/O de arquivos

# In[ ]:


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

# In[ ]:


import numpy as np # Importando a biblioteca numpy e definindo-a com o codnome de np


# In[ ]:


print(np.arange.__doc__) # É sempre possível checar a documentação de uma dada função


# In[ ]:


a = np.arange(15).reshape(3, 5) # Criando um arranjo com 15 elementos e o redimensionando para o formato 3x5


# In[ ]:


a # Escrevendo a


# In[ ]:


a.shape # Verificando as dimensões do arranjo


# In[ ]:


a.ndim # O número de dimensões


# In[ ]:


a.dtype.name # Classificação quando ao tipo dos elementos


# In[ ]:


a.itemsize # Tamanho em bytes de cada elemento


# In[ ]:


a.size # Número total de elementos no arranjo


# In[ ]:


type(a)


# In[ ]:


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
# * Dados tabulares com colunas de tipos heterogêneos, como em uma **tabela SQL ou planilha do Excel**;
# * Dados de **séries temporais** ordenados e não ordenados (não necessariamente de frequência fixa);
# * Dados de matriz arbitrária (homogeneamente digitados ou heterogêneos) com rótulos de linha e coluna;
# * Qualquer outra forma de conjuntos de dados observacionais / estatísticos. Os dados realmente não precisam ser rotulados para serem colocados em uma estrutura de dados de pandas.

# In[ ]:


import pandas as pd


# In[ ]:


df2 = pd.DataFrame({'A': 1.,
                    'B': pd.Timestamp('20130102'),
                    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                    'D': np.array([3] * 4, dtype='int32'),
                    'E': pd.Categorical(["test", "train", "test", "train"]),
                    'F': 'foo'})


# In[ ]:


df2


# In[ ]:


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

# In[ ]:


import sympy as sm
sm.init_printing() #Para escrever equações na tela


# In[ ]:


x, t = sm.symbols('x t') # Criando símbolo


# calcular $\int (e^x \sin(x) + e^x \cos(x)) dx$

# In[ ]:


sm.integrate(sm.exp(x)*sm.sin(x) + sm.exp(x)*sm.cos(x), x)


# calcular a derivada de $\sin(x)e^x$

# In[ ]:


sm.diff(sm.sin(x)*sm.exp(x), x)


# calcular $\int_{-\infty}^{\infty} \sin(x^2)$
# 

# In[ ]:


sm.integrate(sm.sin(x**2), (x, -sm.oo, sm.oo))


# calcular $\lim_{x \to 0} \dfrac{\sin(x)}{x}$

# In[ ]:


sm.limit(sm.sin(x)/x, x, 0)


# resolver $x^2 - 2 = 0$

# In[ ]:


sm.solve(x**2 - 2, x)


# resolver a equação diferencial $y'' - y = e^t$

# In[ ]:


y = sm.Function('y')
eq1 = sm.dsolve(sm.Eq(y(t).diff(t, t) - y(t), sm.exp(t)), y(t))
eq1


# In[ ]:


#Bônus
print(sm.latex(eq1))


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

# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


#Definindo um novo estilo para as figuras [opcional]
plt.style.use(['seaborn-notebook']) 


# In[ ]:


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

# In[ ]:


import bokeh as bk


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

# In[ ]:


x = np.linspace(0., 2*np.pi, num=200, endpoint=True)

sinx = np.sin(x)
cosx = np.cos(x)

plt.plot(x, sinx, label='sin')
plt.plot(x, cosx, label='sin')

plt.legend()
plt.show()


# In[ ]:


#Ou ainda:
for f in [np.sin, np.cos]:
    plt.plot(x, f(x), label=f.__name__)
    
plt.legend()
plt.show()


# In[ ]:


del a, cosx, df2, eq1, sinx, t, x, y
plt.close('all')


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

# In[ ]:


x = np.linspace(0., 2.*np.pi, num=200, endpoint=False)

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
# def ordem2(f=2*pi, i=0., n=7, P=False):
#     '''ordem2(f=2*pi, i=0., n=7, P=False)
# 
#     Calcula os operadores diferenciais da primeira e segunda derivada, para uma
#     malha equidistante de i a f, com n pontos, e erro da ordem de h**2.
# 
#     Parâmetros
#     ----------
#     f : real
#         Valor do contorno superior de x.
#         Padrão é 2*pi.
#     i : real
#         Valor do contorno inferior de x.
#         Padrão é zero.
#     n : inteiro
#         Número de pontos da malha na direção x.
#         Padrão é 7.
#     P : bool, opcional
#         Define que a condição de contorno é periódica quando True.
#         Padrão é False.
# 
#     Retorna
#     -------
#     x, Dx e Dx2, respectivamente o vetor posição e os operadores diferenciais
#     para primeira e segunda ordem.
#     '''
#     #Vetor posição
#     x = linspace(i, f, num=n, endpoint=not P)
#     #Operador diferencial de primeira ordem
#     Dx = diags([-1., 0., 1.],
#               offsets=[-1, 0, 1],
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
#         Dx[-1,-3], Dx[-1,-2], Dx[-1,-1] = 1., -4., 3.
#         Dx2[0,0], Dx2[0,1], Dx2[0,2] = 1., -2., 1.
#         Dx2[-1,-3], Dx2[-1,-2], Dx2[-1,-1] = 1., -2., 1.
#     #
#     h = (x[1]-x[0])
#     Dx /= 2.*h
#     Dx2 /= h**2.
#     return x, Dx, Dx2
# ```

# In[ ]:


#Agora pode-se importar o novo módulo com
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

# In[ ]:


x = np.linspace(0., 2*np.pi, num=201, endpoint=True)

f, dx = np.sin(x), (x[1]-x[0])

Ix = dx*sps.eye(x.size).toarray() #Operador integral

for i in [0, -1]: #Condições de contorno
    Ix[i] *= 0.5

Ix.dot(f).sum() #Integral


# In[ ]:


#Ou alternativamente, utilizando o scipy:
import scipy.integrate
sp.integrate.trapz(f,x)


# Material complementar:
# * [Finite Difference Coefficients Calculator](http://web.media.mit.edu/~crtaylor/calculator.html)
# * [Trapezoidal rule](https://en.wikipedia.org/wiki/Trapezoidal_rule)
# * MOIN, Parviz. **Fundamentals of engineering numerical analysis**. Cambridge University Press, 2010.

# In[ ]:


del D, Dx, Dx2, Ix, dx, f, i, x
plt.close('all')


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

# In[ ]:


k, t1, t2, l, epsi, sigma, tviz, too, h = sm.symbols("k t_1 t_2 L \epsilon \sigma t_{viz} t_\infty h")

eq1 = sm.Eq(k*(t1-t2)/l,epsi*sigma*(t2**4.-tviz**4.)+h*(t2-too))

#Dicionário com os valores que serão substituidos na eq1
dic = {k: .65, t1: 21.+273., l: .25, epsi: .8, sigma: 5.67e-8, tviz: 255., too: 4.+273., h: 23.}

sol = sm.solve(eq1.subs(dic), t2)

for val in sol:
    if val.is_real: #A solução deve ser real
        if val.is_positive: #E positiva
            print('T2 = {0:3.2f} graus celsius'.format(val-273.))  


# In[ ]:


del k, t1, t2, l, epsi, sigma, tviz, too, h, eq1, dic, sol, val


# 2. **Condição de calor transiente bidimensional**. Uma placa de cobre de $50cm \times 50cm$ inicialmente possui temperatura em toda a sua extensão igual a $0^oC$. Instantaneamente , suas bordas são levadas às temperaturas de $60^oC$ em $x=0$; $20^oC$ em $x=50cm$; $0^oC$ em $y=0$ e $100^oC$ em $y=50$. A difusividade térmica do cobre é $1,1532cm^2/s$. Considerando um $\Delta t = 4s$, $\Delta x = \Delta y = 5cm$, calcule a evolução da temperatura para a posição central da placa até o tempo de $400s$. Para o tempo de $200s$ apresente o perfil de temperatura em todos os pontos discretos do domínio.
#     Equação bidimensional:
# \begin{equation}
# \alpha \left( \dfrac{\partial ^2 T}{\partial x^2} + \dfrac{\partial ^2 T}{\partial y^2} \right) =\dfrac{\partial T}{\partial t}
# \end{equation}
#     Discretizando com a derivada segunda numa representação por diferença central e a derivada primeira com diferença ascendente:
# 
# \begin{equation}
# \dfrac{T^{n+1}_{l,j}-T^{n}_{l,j}}{\Delta t}=\alpha \left[ \dfrac{T^{n}_{l-1,j}-2T^{n}_{l,j}+T^{n}_{l+1,j}}{(\Delta x)^2} +\dfrac{T^{n}_{l,j-1}-2T^{n}_{l,j}+T^{n}_{l,j+1}}{(\Delta y)^2}  \right]
# \end{equation}

# In[ ]:


x = np.linspace(0., 50., num=11, endpoint=True)
y = np.linspace(0., 50., num=11, endpoint=True)
t = np.linspace(0., 400., num=101, endpoint=True)

a = 1.1532

T = np.zeros((x.size,y.size,t.size))

T[0,:,:], T[-1,:,:], T[:,0,:], T[:,-1,:] = 60., 20., 0., 100.


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


del T, x, y, t, a, dx2, dy2, dt, i, n, ax, fig, j


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
# Discretizando com as derivadas espaciais numa representação por diferença central e a derivada temporal com diferença ascendente:
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
# 
# onde $D_x$ e $D_x^2$ são os operadores diferenciais de primeira e segunda ordem, respectivamente.

# In[ ]:


x, Dx, Dx2 = dv.ordem2(1., n=101)

t = np.linspace(0., 8., num=8001, endpoint=True)

dt = t[1]-t[0]

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

    A = dt*(alpha*Dx2-u*Dx)
    
    return A, T


# In[ ]:


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


del A,D,Dx,Dx2,T,alpha,ax,dt,fig,i,j,n,t,u,visu,x
plt.close('all')


# ### Vibrações Mecânicas
# 
# <img src="molas.png" width="480">
# 
# 1. O movimento das massas esquematizadas na figura acima é governado pelo seguinte sistema de equações:
# 
# \begin{equation}
# \underbrace{
# \begin{bmatrix}
# m_0 & 0 \\
#   0 & m_1
# \end{bmatrix}
# }_{M}
# \begin{bmatrix} \ddot x_0 \\ \ddot x_1 \end{bmatrix} +
# \underbrace{
# \begin{bmatrix}
# c_0 + c_1 & -c_1 \\
# -c_1 & c_1 + c_2
# \end{bmatrix}
# }_{C}
# \begin{bmatrix} \dot x_0 \\ \dot x_1 \end{bmatrix} +
# \underbrace{
# \begin{bmatrix}
# k_0 + k_1 & -k_1 \\
# -k_1 & k_1 + k_2
# \end{bmatrix}
# }_{K}
# \begin{bmatrix} x_0 \\ x_1 \end{bmatrix} = 0
# \end{equation}
# 
#     No caso particular em que o coeficiente de amortecimento é nulo $c=0$, a resposta esperada do sistema pode ser escrita como $x = a e^{iwt}$, que substituida no sistema acima resulta em:
# 
# \begin{equation}
#     \left[ \underbrace{M^{-1}K}_{A} - \omega^2I \right] x = 0
# \end{equation}
# 
#     Sabendo que as frequencias $\omega$ e modos de vibração $a$ catacterísticos do sistema podem ser obtidos respectivamente como os autovalores e autovetores da matriz $A$, grafique a resposta em função do tempo considerando $m_0=m_1=1$ e $k_0=k_1=k_2=1$.

# In[ ]:


m0, m1 = 1., 1.
k0, k1, k2 = 1., 1., 1.

M = np.array([[m0, 0.],
              [0., m1]])
K = np.array([[k0+k1, -k1  ],
              [-k1  , k1+k2]])

from scipy import linalg
A = sp.linalg.inv(M).dot(K)

e_vals, e_vecs = sp.linalg.eig(A)


# In[ ]:


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


# In[ ]:


del A,K,M,a,ax,e_vals,e_vecs,fig,i,j,k0,k1,k2,m0,m1,offset,t,w
plt.close('all')


# ### Engenharia Econômica

# ### Eletrônica
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

# In[ ]:


r1, r2, r3, r4, r5 = 10., 22., 47., 22., 10.
v1, v2 = 9., 9.

A = [[1., 1., -1.],
     [r1+r4, 0., r3],
     [0., r2+r5, r3]]

B = [0., v1, v2]


# In[ ]:


I = np.linalg.solve(A,B)

for i, ival in enumerate(I):
    print('I{} = {}'.format(i, ival))


# Material complementar:
# * [Kirchhoff's circuit laws](https://en.wikipedia.org/wiki/Kirchhoff%27s_circuit_laws)

# In[ ]:


del A,B,I,i,ival,r1,r2,r3,r4,r5,v1,v2


# ### Resistência dos Materiais
# 
# 1. Determine os esforços am cada uma das barras que compôem a treliça abaixo. Para cada um dos nós observa-se que $\sum F_x = 0$ e $\sum F_y = 0$ devem ser satisfeitas, resultando em 14 equações e um sistema matricial $14 \times 14$ a ser resolvido. Como incógnitas, tem-se a força resultante nas 11 barras $F_{ij}$, além da força nos apoios: $F_{1x}$, $F_{1y}$ e $F_{4y}$. Considerar o carregamento como $F_5 = F_6 = 1kN$ e $F_7 = 2kN$.
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

# In[ ]:


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


# In[ ]:


I = np.linalg.solve(A,B)

for i, key in enumerate(['F1x','F1y','F4y','F15','F12','F23','F25','F27','F34','F36','F37','F46','F57','F67']):
    print('{0:2d}: {1} = {2:5.2f}kN'.format(i,key,I[i]))


# In[ ]:


del A,B,F5,F6,F7,I,angle,cos,i,key,sin


# ## Exercícios Propostos

# 1. **Ciência dos Materiais - Difusão combinada de calor e massa em um caso bidimensional**.
# 
#     Uma chapa de aço, de espessura $L$, inicialmente na temperatura $T_i$, entra em um forno para tratamento térmico. Ao contato com a atmosfera aquecida e rica em cardono dentro do forno, os contornos da peça atingem temperatura e concentração de $T_p$ e $c_p$, respectivamente. Sabendo que as equações unidimensionais para a difusão de calor e de concentração de carbono são escritas em sua forma adimensional como:
# \begin{equation}
#     \dfrac{\partial ^2 \Theta}{\partial X^2} = \dfrac{\partial \Theta}{\partial \tau},
# \end{equation}
# \begin{equation}
#     \dfrac{\partial }{\partial X} \left( \dfrac{1}{Le(\Theta)} \dfrac{\partial C}{\partial X} \right)  = \dfrac{\partial C}{\partial \tau},
# \end{equation}
# onde $\Theta(X, \tau) = \dfrac{T(x,t) - T_i}{T_p - T_i}$ representa a temperatura adimensional, $X = \dfrac{x}{L}$ o sistema de coordenadas, $\tau = \dfrac{\alpha t}{L^2}$ o tempo, $C(X, \tau) = \dfrac{c(x,t) - c_i}{c_p - c_i}$ a concentração de carbono e $Le = \dfrac{\alpha}{D}$ o número de Lewis. Determine o tempo adimensional necessário para que no centro da peça $C(X/2)=75\%$. Grafique a evolução da temperatura e da concentração no centro da peça em função do tempo. Em situações reais, a difusão mássica é muito menor que a diffusão térmica, de modo que o número de Lewis seja da ordem de $10^6$, desta maneira, para fins didáticos, assuma que $Le = \dfrac{4}{1+2\Theta^{3/2}}$.

# 2. **Visualizações animadas**. Pode-se tirar proveito de belas animações interativas produzidas pela Matplotlib. Para tanto, dois exercícios resolvidos podem ser revisitados:
# 
#     a. Primeiramente, o exercício 3.3.1 de vibrações mecânicas, produzindo uma animação para a variação de posição das duas massas com o tempo;
# 
#     b. O problema 3.2.2 de fenômenos de transporte, gerando uma uma animação para a evolução temporal da temperatura na placa.
# 
#     Material complementar:
#     * [Embedding Matplotlib Animations in Jupyter as Interactive JavaScript Widgets](http://louistiao.me/posts/notebooks/embedding-matplotlib-animations-in-jupyter-as-interactive-javascript-widgets/)

# 3. **Resistência dos Materiais**. De volta ao exercício resolvido 3.6.1, encontre um maneira de produzir as matrizes `A` e `B` automaticamente, sendo fornecidos ao código apenas a posição espacial dos nós da treliça, e as forças de ligação entre eles. Compare com a resposta obtida anteriormente para verificação do novo código. Grafique os resultados.

# 4. **Fenômenos de Transporte**. Empregar o conceito de operador diferencial estabelecido pelo módulo `diferencial.py` ao exercício resolvido 3.2.2. Comparar com a resposta obtida anteriormente para verificação do novo código.
