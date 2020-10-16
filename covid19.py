# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# importando as bibliotecas que estaremos usando no trabalho
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest


# %%
# importando o dataset
covid = pd.read_csv(r'caso.csv', encoding = "utf-8")
covid.describe()


# %%
# vamos começar pelo pre processamento de dados


# %%
# olhando para o tamanho do dataset vemos que temos 441580 linhas e 12 colunas 
# mas estaremos usando apenas uma fração desses dados, Recife e São Paulo 
# também adicionaremos outras colunas que acharmos necessário
covid.shape


# %%
# como ha dados com tipos object, o que nao pode ser bom pra categorizar, vamos mudar pra tipo categorico
covid.dtypes


# %%
# ajustando tipos dos dados
covid['date'] = covid['date'].astype('category') 
covid['state'] = covid['state'].astype('category') 
covid['city'] = covid['city'].astype('category') 
covid['place_type'] = covid['place_type'].astype('category') 
covid.dtypes


# %%
covid['state'].cat.categories


# %%
# o nosso próximo passo eh ver a quantidade de elementos nulos em cada coluna e podemos observar que ha 4 colunas bastante dados nulos 
print(covid.isnull().sum())


# %%
covidRecife = covid[covid.city=='Recife']
covidRecife.isnull().sum()


# %%
covidSP = covid[covid.city=='São Paulo']
covidSP.isnull().sum()


# %%
# como vimos acima que os dados que iremos trabalhar nao tem nenhum dado ausente, decidimos por limpar o dataset original com instâncias com ao menos 1 coluna com dado ausente
covid = covid.dropna()
covid.shape


# %%
# Normalização

# min-max do scikit
scaler = MinMaxScaler()
covidRecife['confirmed_norm'] = scaler.fit_transform(covidRecife[['confirmed']])
covidRecife['confirmed_norm'].describe()


# %%
# Discretização

# discretizando com 10 bins com mesmo intervalo
covidRecife['deaths_dist'] = pd.cut(covidRecife['deaths'],4)


# %%
covidRecife['deaths_dist'].value_counts()


# %%
covidRecife['deaths'].describe()


# %%
# uma estatística importante no nosso trabalho eh o número de mortes por dia, mas isso nao tem no dataset, por isso criamos esta função que adiciona esta coluna usando o metodo diff

# pegando os dados de Recife, resetando o index e adicionando as colunas
covidRecife = covidRecife.sort_values(by=['date'], ascending=True)
covidRecife.reset_index(inplace=True, drop=True)

# utilizando o diff() do pandas para calcular a diferença entre linhas consecutivas
covidRecife['deaths_per_day'] = covidRecife.deaths.diff().abs()

# aqui eu imputo o dado pois o resultado da função diff() executada acima faz com que tenhamos um NaN na primeira linha da nova coluna e para nao perder os dados do primeiro dia não utilizamos o dropna() aqui e nem nas próximas colunas criadas da mesma forma. 
covidRecife['deaths_per_day'][0] = 0

covidRecife.head()


# %%
# pegando  dados de São Paulo, resetando o index e adicionando as colunas
covidSP = covid[covid.city == 'São Paulo'].sort_values(by=['date'], ascending=True)
covidSP.reset_index(inplace=True, drop=True)
covidSP['deaths_per_day'] = covidSP.deaths.diff().abs()

# aqui eu imputo o dado para nao perder os dados do primeiro dia
covidSP['deaths_per_day'][0] = 0
covidSP.head()


# %%
# não eh uma boa estrategia comparar o número de mortes bruto em duas cidades com quantidades de habitantes discrepantes
# por isso vamos usar a função apply do pandas para ter número de mortes a cada 100 mil de habitantes

print("População de Recife: ", covidRecife.estimated_population_2019[0])
print("População de São Paulo: ", covidSP.estimated_population_2019[0])


# %%

# adicionando coluna do total de mortes por 100 mil habitantes nos dataframes de Recife e São Paulo
covidRecife['deaths_per_100k_inhabitants'] = covidRecife.apply(lambda x: x['deaths']/(x['estimated_population_2019'] / 100000), axis=1).abs()

covidSP['deaths_per_100k_inhabitants'] = covidSP.apply(lambda x: x['deaths']/(x['estimated_population_2019'] / 100000), axis=1).abs()

# tratando indice 0 com NaN
covidRecife['deaths_per_100k_inhabitants'][0] = 0
covidSP['deaths_per_100k_inhabitants'][0] = 0


# %%
# adicionando coluna de mortes por dia por 100 mil habitantes nos dataframes de Recife e São Paulo
covidRecife['deaths_per_day_per_100k_inhabitants'] = covidRecife.apply(lambda x: x['deaths_per_day']/(x['estimated_population_2019'] / 100000), axis=1).abs()

covidSP['deaths_per_day_per_100k_inhabitants'] = covidSP.apply(lambda x: x['deaths_per_day']/(x['estimated_population_2019'] / 100000), axis=1).abs()

# tratando indice 0 com NaN
covidRecife['deaths_per_day_per_100k_inhabitants'][0] = 0
covidSP['deaths_per_day_per_100k_inhabitants'][0] = 0


# %%
# adicionando coluna de casos confirmados por dia nos dataframes de Recife e São Paulo
covidRecife['confirmed_per_day'] = covidRecife.confirmed.diff().abs()
covidSP['confirmed_per_day'] = covidSP.confirmed.diff().abs()

# adicionando coluna de casos confirmados por dia por dia por 100 mil habitantes nos dataframes de Recife e São Paulo
covidRecife['confirmed_per_day_per_100k_inhabitants'] = covidRecife.apply(lambda x: x['confirmed_per_day']/(x['estimated_population_2019'] / 100000), axis=1).abs()
covidSP['confirmed_per_day_per_100k_inhabitants'] = covidSP.apply(lambda x: x['confirmed_per_day']/(x['estimated_population_2019'] / 100000), axis=1).abs()

# tratando indice 0 com NaN
covidRecife['confirmed_per_day'][0] = 0
covidSP['confirmed_per_day'][0] = 0

covidRecife['confirmed_per_day_per_100k_inhabitants'][0] = 0
covidSP['confirmed_per_day_per_100k_inhabitants'][0] = 0

covidSP.describe()


# %%
covidRecife.describe()


# %%
# o nosso objetivo neste trabalho eh analisar especificamente as cidade de Recife e São Paulo, e como vimos, os dados nao apresentam falha com relação a elementos nulos, por isso a imputação de dados não vai ser necessaria. 

# sobre detecção de outliers ...


# %%
crec_confirmed_per_day = covidRecife.confirmed_per_day[0:-1:1]
csp_confirmed_per_day = covidSP.confirmed_per_day[0:-1:1]
plot_confirmed_per_day_box = pd.concat([crec_confirmed_per_day, csp_confirmed_per_day], axis=1, keys=['Recife', 'Sao Paulo'])
plot_confirmed_per_day_box.plot(kind='box', figsize=[10, 8])
plt.ylabel('Casos Confirmados')
plt.xlabel('Cidades')

# a mesma tendencia segue no número de mortes

# comparação entre o número de mortes total em Recife (azul) e São Paulo (laranja)
crec_deaths_per_day = covidRecife.deaths_per_day[0:-1:1]
csp_deaths_per_day = covidSP.deaths_per_day[0:-1:1]
plot_deaths_per_day_box = pd.concat([crec_deaths_per_day, csp_deaths_per_day], axis=1, keys=['Recife', 'Sao Paulo'])
plot_deaths_per_day_box.plot(kind='box', figsize=[10, 8])
plt.ylabel('Mortes')
plt.xlabel('Cidades')


# %%
covidRecife['confirmed_per_day'].plot(kind="hist", figsize=(10,5))
covidSP['confirmed_per_day'].plot(kind="hist", figsize=(10,5))


# %%
covidSP.head()


# %%
# Detecção de Outliers

covidRecife1 = covidRecife.drop(columns=['date', 'state', 'city', 'place_type', 'is_last'], axis=1)
del covidRecife1['deaths_dist']


# %%
rng = np.random.RandomState(42)
clf = IsolationForest(max_samples=100, random_state=rng)
clf.fit(covidRecife1)


# %%
scores = clf.predict(covidRecife1)


# %%
scores


# %%
covidRecife1['outlier'] = scores


# %%
covidRecife1 = covidRecife1[covidRecife1['outlier'] != -1]
len(covidRecife1)


# %%
covidRecife1.describe()


# %%
covidRecife['deaths_per_day'].plot(kind="bar", figsize=(20,5))


# %%
covidSP['deaths_per_day'].plot(kind="bar", figsize=(20,5))


# %%
# comparação entre o número de casos confirmados totais em Recife (azul) e São Paulo (laranja)
crec_confirmed = covidRecife.confirmed[0:-1:1]
csp_confirmed = covidSP.confirmed[0:-1:1]
plot_confirmed = pd.concat([crec_confirmed, csp_confirmed], axis=1, keys=['Recife', 'Sao Paulo'])
plot_confirmed.plot(kind='line', figsize=[6, 4])
plt.ylabel('Casos Confirmados')
plt.xlabel('dias')

# a mesma tendencia segue no número de mortes

# comparação entre o número de mortes total em Recife (azul) e São Paulo (laranja)
crec_deaths = covidRecife.deaths[0:-1:1]
csp_deaths = covidSP.deaths[0:-1:1]
plot_deaths = pd.concat([crec_deaths, csp_deaths], axis=1, keys=['Recife', 'Sao Paulo'])
plot_deaths.plot(kind='line', figsize=[6, 4])
plt.ylabel('Mortes')
plt.xlabel('dias')

# no grafico abaixo a cidade de São Paulo parece ter sido muito mais afetada do que Recife, pois temos uma curva de mortes muito maior


# %%
# Veremos agora o comparativo entre casos confirmados por 100 mil habitantes e mortes por 100 mil habitantes entre Recife e São Paulo

# comparação entre o número de casos confirmados por 100 mil habitantes em Recife (azul) e São Paulo (laranja)
crec_confirmed_per_100k_inhabitants = covidRecife.confirmed_per_100k_inhabitants[0:-1:1]
csp_confirmed_per_100k_inhabitants = covidSP.confirmed_per_100k_inhabitants[0:-1:1]
plot_confirmed_per_100k_inhabitants = pd.concat([crec_confirmed_per_100k_inhabitants, csp_confirmed_per_100k_inhabitants], axis=1, keys=['Recife', 'Sao Paulo'])
plot_confirmed_per_100k_inhabitants.plot(kind='line', figsize=[6, 4])
plt.ylabel('Casos Confirmados por 100 mil Habitantes')
plt.xlabel('dias')

# a mesma tendencia segue no número de mortes

# comparação entre o número de mortes por 100 mil habitantes em Recife (azul) e São Paulo (laranja)
crec_deaths_per_100k_inhabitants = covidRecife.deaths_per_100k_inhabitants[0:-1:1]
csp_deaths_per_100k_inhabitants = covidSP.deaths_per_100k_inhabitants[0:-1:1]
plot_deaths_per_100k_inhabitants = pd.concat([crec_deaths_per_100k_inhabitants, csp_deaths_per_100k_inhabitants], axis=1, keys=['Recife', 'Sao Paulo'])
plot_deaths_per_100k_inhabitants.plot(kind='line', figsize=[6, 4])
plt.ylabel('Mortes por 100 mil Habitantes')
plt.xlabel('dias')

# no gráfico abaixo a cidade de Recife parece ter sido muito mais afetada do que São Paulo, pois temos uma curva maior em relação a casos confirmados/mortes por 100 mil habitantes


# %%

# comparação entre o número de casos confirmados por dia por 100 mil habitantes em Recife (azul) e São Paulo (laranja)
crec_confirmed_per_day_per_100k_inhabitants = covidRecife.confirmed_per_day_per_100k_inhabitants[0:-1:1]
csp_confirmed_per_day_per_100k_inhabitants = covidSP.confirmed_per_day_per_100k_inhabitants[0:-1:1]
plot_confirmed_per_day_per_100k_inhabitants = pd.concat([crec_confirmed_per_day_per_100k_inhabitants, csp_confirmed_per_day_per_100k_inhabitants], axis=1, keys=['Recife', 'Sao Paulo'])
plot_confirmed_per_day_per_100k_inhabitants.plot(kind='line', figsize=[6, 4])
plt.ylabel('Casos Confirmados por dia por 100 mil Habitantes')
plt.xlabel('dias')

# a mesma tendencia segue no número de mortes

# comparação entre o número de mortes por dia por 100 mil habitantes em Recife (azul) e São Paulo (laranja)
crec_deaths_per_day_per_100k_inhabitants = covidRecife.deaths_per_day_per_100k_inhabitants[0:-1:1]
csp_deaths_per_day_per_100k_inhabitants = covidSP.deaths_per_day_per_100k_inhabitants[0:-1:1]
plot_deaths_per_day_per_100k_inhabitants = pd.concat([crec_deaths_per_day_per_100k_inhabitants, csp_deaths_per_day_per_100k_inhabitants], axis=1, keys=['Recife', 'Sao Paulo'])
plot_deaths_per_day_per_100k_inhabitants.plot(kind='line', figsize=[6, 4])
plt.ylabel('Mortes por por dia 100 mil Habitantes')
plt.xlabel('dias')

# no gráfico abaixo a cidade de Recife parece ter sido muito mais afetada do que São Paulo, pois temos uma curva maior em relação a casos confirmados/mortes por 100 mil habitantes


# %%
# Aqui fazemos a correlação entre as colunas de covidRecife
covidRecife.corr(method='spearman')


# %%
fig = plt.figure()
ax = plt.axes(projection="3d")

plt.show()


# %%



# %%

fig = plt.figure()
ax = plt.axes(projection="3d")

z_line = np.linspace(0, 15, 1000)
x_line = np.cos(z_line)
y_line = np.sin(z_line)
ax.plot3D(x_line, y_line, z_line, 'gray')

z_points = 15 * np.random.random(100)
x_points = np.cos(z_points) + 0.1 * np.random.randn(100)
y_points = np.sin(z_points) + 0.1 * np.random.randn(100)
ax.scatter3D(x_points, y_points, z_points, c=z_points, cmap='hsv');


plt.show()


# %%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x =[covidRecife.confirmed]
y =[covidRecife.deaths]
z =[covidRecife.order_for_place]

ax.scatter(x, y, z, c='r', marker='o')

ax.set_xlabel('Casos confirmados')
ax.set_ylabel('Mortes')
ax.set_zlabel('Dias')

plt.show()


# %%
# aqui eu reuno as informações sobre mortes por dia de ambas as cidades
crec_dp100k = covidRecife.deaths_per_day_per_100k_inhabitants[0:136:1]
csp_dp100k = covidSP.deaths_per_day_per_100k_inhabitants[0:136:1]


# %%
# aqui eu concateno ambos os DataFrames e coloco o nome de cada box usando o keys do concat
df_rec_sp = pd.concat([crec_dp100k, csp_dp100k], axis=1, keys=['deaths_per_day_per_100k_in_Recife', 'deaths_per_day_per_100k_in_Sao_Paulo'])


# %%
df_rec_sp.plot(kind='box', figsize=[10, 5])


# %%
# verificando se a diferença segue a gaussiana
# df_rec_sp['dpdp100k_difference'] = df_rec_sp.apply(lambda x: x['deaths_per_day_per_100k_in_Recife'] - x['deaths_per_day_per_100k_in_Sao_Paulo'], axis=1) or :
df_rec_sp['dpdp100k_difference'] = df_rec_sp['deaths_per_day_per_100k_in_Recife'] - df_rec_sp['deaths_per_day_per_100k_in_Sao_Paulo']
print(df_rec_sp)


# %%
df_rec_sp['dpdp100k_difference'].plot(kind='hist')


# %%
# Shapiro-Wilk teste de normalidade que retorna o seguinte parametro: (valor, p-valor). Hipótese Nula: as mortes em Recife não são normalmente distribuídas.
stats.shapiro(df_rec_sp['dpdp100k_difference'])

# um p-valor menor que o valor crítico indica que a hipótese nula foi rejeitada / shapiro => retorn (valor crítico, p-valor)


# %%
# checando simetria
df_rec_sp['dpdp100k_difference'].plot(kind='box')


