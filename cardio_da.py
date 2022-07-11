import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D


def get_data(path):
    # This function get the dataset
    return pd.read_csv(path, sep=';')


def age_to_years(data):
    # Age in dataset is set to days here we transform to years.
    data['age'] = data['age'] / 365
    # Float -> Int (removal of decimal)
    data['age'] = data['age'].astype(np.int64)

def cardio_mod(data):
    data.loc[data['cardio'] == 1, 'cardio'] = 'Doente'
    data.loc[data['cardio'] == 0, 'cardio'] = 'Saudavel'

def create_pd(data, min_age, max_age):
    # Creating a new DF, so we can calculate the probability of each age group be healthy or sick.

    dc = {'age': [], 'sick': [], 'wealth': [], 'total_person': []}
    for a in range(min_age, max_age + 1, 1):
        total_person_at_age = data[data['age'] == a]['age'].shape[0]
        total_sick = data[(data['age'] == a) & (data['cardio'] == 1)]['age'].shape[0]
        total_wealth = data[(data['age'] == a) & (data['cardio'] == 0)]['age'].shape[0]
        tp, ts, tw = total_person_at_age, total_sick, total_wealth
        dc['age'].append(a)
        dc['total_person'].append(tp)
        dc['sick'].append((ts / float(tp)) * 100 if tp > 0 else 0)
        dc['wealth'].append((tw / float(tp)) * 100 if tp > 0 else 0)
    return pd.DataFrame.from_dict(dc)


def prob_vis(df):
    # Relative Probability Visualization
    ax = sns.lineplot(x=df['age'], y=df['sick'], color="red")
    ax.set(xlabel='Idade', ylabel='Probabilidade de estar doente (em %)', ylim=(0, 100))
    ax1 = sns.lineplot(x=df['age'], y=df['wealth'], color="blue", ax=ax.axes.twinx())
    ax1.set(xlabel='Idade', ylabel='Probabilidade de estar saudável (em %)', ylim=(0, 100))
    ax.legend(handles=[Line2D([], [], marker='', color="red", label='Doentes'),
                       Line2D([], [], marker='', color="b", label='Saudáveis')])
    plt.show()


def imc_and_cat(data):
    # Creation of BMI column and its categories where:
    # < 18.5 - Underweight (1)
    # Between 18.5 and 24.9 - Normal Weight(2)
    # Between 25 and 29.9 - Pre-obesity(3)
    # Between 30 and 34.9 - Obesity Class I(4)
    # Between 35 and 39.9 - Obesity Class II(5)
    # > 40 - Obesity Class III(6)
    # Source: World Health Organization

    data['bmi'] = np.round(data['weight'] / (data['height'] / 100) ** 2.2)
    data['bmi_cat'] = data['bmi'].apply(lambda x: 1 if x < 18.5
                                        else 2 if (x >= 18.5) & (x <= 24.9)
                                        else 3 if (x >= 25) & (x <= 29.9)
                                        else 4 if (x >= 30) & (x <= 34.9)
                                        else 5 if (x >= 35) & (x <= 39.9) else 6)
    return data['bmi'], data['bmi_cat']


def imc_graph(data):
    ax = sns.histplot(x=data['bmi_cat'], hue=data['cardio'], kde=True, bins=10, multiple='dodge')
    ax.set(xlabel='Categoria IMC')
    plt.show()


def gender_mod(data):
    data.loc[data['gender'] == 1, 'gender'] = 'Female'
    data.loc[data['gender'] == 2, 'gender'] = 'Male'


def sex_graph(data):
    ax = sns.countplot(data=data, x='gender', hue='cardio')
    ax.set(xlabel = 'Sexo')
    plt.show()

if __name__ == '__main__':
    data = get_data(path='cardio_train.csv')

    age_to_years(data)

    df = create_pd(data, data['age'].min(), data['age'].max())

    prob_vis(df)

    data['bmi'], data['bmi_cat'] = imc_and_cat(data)

    imc_graph(data)

    gender_mod(data)

    cardio_mod(data)

    sex_graph(data)