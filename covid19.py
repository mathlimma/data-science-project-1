import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

covid = pd.read_csv(r'caso.csv', encoding = "utf-8")

#primeiro ordenaremos pela data de forma ascendente (primeiro dia ao ultimo)
covidRecife = covid[covid.city == 'Recife'].sort_values(by=['date'], ascending=True)

#resetaremos o índice da tabela original e apresentaremos o índice da tabela almejada
covidRecife.reset_index(inplace=True, drop=True)
covidRecife.tail()

#apresentação do gráfico da quantidade de mortes total do dia
plt.plot(covidRecife['order_for_place'], (covidRecife['deaths']))

#first try with a for loop
def sub_column_value(newcolumn, subcolumn, table):
    table[newcolumn] = 0
    for index, i in enumerate(subcolumn):
        if index > 0:
            table[newcolumn][index] = (subcolumn[index] - subcolumn[index-1])

sub_column_value('deaths_per_day', covidRecife.deaths, covidRecife)
         
covidRecife.tail()

#death_per_day bar graph
covidRecife['deaths_per_day'].plot.bar(figsize=(30,10))
