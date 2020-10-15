#start by installing python through pyenv (use version 3.5.6 for scipy sake)=> https://realpython.com/intro-to-pyenv/
#scipy and other packages => https://www.scipy.org/install.html#pip-install

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats

# importando o dataset
covid = pd.read_csv(r'caso.csv', encoding = "utf-8")
covid.describe()


# vamos começar pelo pre processamento de dados

# olhando para o tamanho do dataset vemos que temos 441580 linhas e 12 colunas 
# mas estaremos usando apenas uma fração desses dados Recife e São Paulo 
# também adicionaremos outras colunas que achamos necessário
covid.shape

# como ha dados com tipos object, o que nao pode ser bom pra categorizar, vamos mudar pra tipo categorico
covid.dtypes

# ajustando tipos dos dados
covid['date'] = covid['date'].astype('category') 
covid['state'] = covid['state'].astype('category') 
covid['city'] = covid['city'].astype('category') 
covid['place_type'] = covid['place_type'].astype('category') 
covid.dtypes

covid['state'].cat.categories

# o nosso próximo passo eh ver a quantidade de elementos nulos em cada coluna e podemos observar que ha 4 colunas bastante dados nulos 
print(covid.isnull().sum())

covidRecife = covid[covid.city=='Recife']
covidRecife.isnull().sum()

covidSP = covid[covid.city=='São Paulo']
covidSP.isnull().sum()

# como vimos acima que os dados que iremos trabalhar nao tem nenhum dado ausente, decidimos por limpar o dataset original com instâncias com ao menos 1 coluna com dado ausente
covid = covid.dropna()
covid.shape

# o nosso objetivo neste trabalho eh analisar especificamente as cidade de Recife e São Paulo, e como vimos, os dados nao apresentam falha com relação a elementos nulos, por isso a imputação de dados não vai ser necessaria. 

# sobre detecção de outliers ...

# uma estatística importante no nosso trabalho eh o número de mortes por dia, mas isso nao tem no dataset, por isso criamos esta função que adiciona esta coluna usando o metodo diff

# pegando os dados de Recife, resetando o index e adicionando as colunas
covidRecife = covidRecife.sort_values(by=['date'], ascending=True)
covidRecife.reset_index(inplace=True, drop=True)

#first try with a for loop
# def sub_column_value(newcolumn, subcolumn, table):
#     table[newcolumn] = 0
#     for index, i in enumerate(subcolumn):
#         if index > 0:
#             value = (subcolumn[index] - subcolumn[index-1])
#             if value > 0:
#                 table[newcolumn][index] = value
#             else:
#                 table[newcolumn][index] = 0

# sub_column_value('deaths_per_day', covidRecife.deaths, covidRecife)

# using pandas diff to calculate the difference between consecutive rows
covidRecife['deaths_per_day'] = covidRecife.deaths.diff()

# aqui eu imputo o dado para nao perder os dados do primeiro dia
covidRecife['deaths_per_day'][0] = 0

covidRecife.head()

# pegando  dados de São Paulo, resetando o index e adicionando as colunas
covidSP = covid[covid.city == 'São Paulo'].sort_values(by=['date'], ascending=True)
covidSP.reset_index(inplace=True, drop=True)
covidSP['deaths_per_day'] = covidSP.deaths.diff()

# aqui eu imputo o dado para nao perder os dados do primeiro dia
covidSP['deaths_per_day'][0] = 0

covidSP.head()

#death_per_day bar graph
covidRecife['deaths_per_day'].plot(kind="bar", figsize=(20,5))

# bar graph of deaths per day in SP
covidSP['deaths_per_day'].plot(kind="bar", figsize=(20,5))

# comparação entre o número de mortes total em Recife (azul) e São Paulo (laranja)
crec_deaths = covidRecife.deaths[0:-1:1]
csp_deaths = covidSP.deaths[0:-1:1]
plot_deaths = pd.concat([crec_deaths, csp_deaths], axis=1, keys=['Recife', 'Sao Paulo'])
plot_deaths.plot(kind='line', figsize=[6, 4])
# no grafico abaixo a cidade de São Paulo parece ter sido muito mais afetada do que Recife, pois temos uma curva de mortes muito maior

# comparação entre o número de casos confirmados totais em Recife (azul) e São Paulo (laranja)
crec_confirmed = covidRecife.confirmed[0:-1:1]
csp_confirmed = covidSP.confirmed[0:-1:1]
plot_confirmed = pd.concat([crec_confirmed, csp_confirmed], axis=1, keys=['Recife', 'Sao Paulo'])
plot_confirmed.plot(kind='line', figsize=[6, 4])
# a mesma tendencia segue no número de casos

# deaths per day per 1 million inhabitants function without for loop 
# def deaths_per_day_per_1kk(newcolumn, table):
#     population_per_1kk = table.estimated_population_2019[0] / 1000000
#     table[newcolumn] = table.deaths_per_day / population_per_1kk
# deaths_per_day_per_1kk('deaths_per_day_per_1kk_inhabitants', covidRecife)

# recurso do pandas: apply
covidRecife['deaths_per_day_per_1kk_inhabitants'] = covidRecife.apply(lambda x: x['deaths_per_day']/(x['estimated_population_2019'] / 1000000), axis=1)
covidSP['deaths_per_day_per_1kk_inhabitants'] = covidSP.apply(lambda x: x['deaths_per_day']/(x['estimated_population_2019'] / 1000000), axis=1)

# comparison graph of deaths per day per 1 million inhabitants between Recife and São Paulo
plt.plot(covidRecife['order_for_place'], covidRecife['deaths_per_day_per_1kk_inhabitants'], label="Recife")
plt.plot(covidSP['order_for_place'], covidSP['deaths_per_day_per_1kk_inhabitants'], label="São Paulo")
# linha azul => recife / linha laranja => SP

# adding 2 more columns: confirmed_per_day and confirmed_per_day_per_1kk_inhabitants
covidRecife['confirmed_per_day'] = covidRecife.confirmed.diff()
covidSP['confirmed_per_day'] = covidSP.confirmed.diff()
covidRecife['confirmed_per_day_per_1kk_inhabitants'] = covidRecife.apply(lambda x: x['confirmed_per_day']/(x['estimated_population_2019'] / 1000000), axis=1)
covidSP['confirmed_per_day_per_1kk_inhabitants'] = covidSP.apply(lambda x: x['confirmed_per_day']/(x['estimated_population_2019'] / 1000000), axis=1)

covidSP.tail()

# statistical data about covidRecife
covidRecife.describe()

# statistical data about covidSP
covidSP.describe()

# boxplot
plt.boxplot([covidRecife.deaths_per_day_per_1kk_inhabitants, covidSP.deaths_per_day_per_1kk_inhabitants], labels=['Recife', 'São Paulo'])

# here i gathered the informations about deaths per day from both cities
crec_dp1kk = covidRecife.deaths_per_day_per_1kk_inhabitants[0:136:1]
csp_dp1kk = covidSP.deaths_per_day_per_1kk_inhabitants[0:136:1]

# here i concatenate both DataFrames and change the name of the tables using keys
df_rec_sp = pd.concat([crec_dp1kk, csp_dp1kk], axis=1, keys=['deaths_per_day_per_1kk_in_Recife', 'deaths_per_day_per_1kk_in_Sao_Paulo'])

df_rec_sp.describe()

df_rec_sp.plot(kind='line', figsize=[10, 5])
df_rec_sp.plot(kind='box', figsize=[10, 5])

# check if the difference follows the gaussian
# df_rec_sp['dpdp1kk_difference'] = df_rec_sp.apply(lambda x: x['deaths_per_day_per_1kk_in_Recife'] - x['deaths_per_day_per_1kk_in_Sao_Paulo'], axis=1),  ou :
df_rec_sp['dpdp1kk_difference'] = df_rec_sp['deaths_per_day_per_1kk_in_Recife'] - df_rec_sp['deaths_per_day_per_1kk_in_Sao_Paulo']

# checando quantas linhas com valores nulos tem a tabela
print(df_rec_sp.isnull().sum())

# a linha com valores NaN dará uma inconsistência no cálculo do teste de hipótese. Por isso precisamos limpar as linhas vazias a seguir, nesse caso apenas a primeira.
df_rec_sp = df_rec_sp.dropna()
print(df_rec_sp.isnull().sum())

# histogram
df_rec_sp['dpdp1kk_difference'].plot(kind='hist')

# Shapiro-Wilk normality test (value, p-value). Null hypothesis: the deaths in recife are not normally distributed.
stats.shapiro(df_rec_sp['dpdp1kk_difference'])
# a p-value less than the critical value indicates that the null hypothesis was rejected / shapiro => return (critical value, p-value)

# executing the t-test
stats.ttest_rel(df_rec_sp['deaths_per_day_per_1kk_in_Recife'], df_rec_sp['deaths_per_day_per_1kk_in_Sao_Paulo'])

# checking symmetry
df_rec_sp['dpdp1kk_difference'].plot(kind='box')

# wilcoxon test
stats.wilcoxon(df_rec_sp['deaths_per_day_per_1kk_in_Recife'], df_rec_sp['deaths_per_day_per_1kk_in_Sao_Paulo'])