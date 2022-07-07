# Projeto para previsão de doenças cardiovasculares

## Motivação
Uma das motivações para esse projeto foi desafiar meus conhecimentos em python e em machine learning, fazendo uma análise sobre os dados presentes no dataset.

## Instalação
Se você deseja rodar o código em seu dispositivo, rode o seguinte comando no seu terminal:
```console
pip install -r requirements.txt
```

## Analise sobre os dados presentes
### Comparando a presença de doença entre homens e mulheres

Este primeiro gráfico trata-se da incidência de doenças cardiovasculares em homens e mulheres.

![sex_graph](https://user-images.githubusercontent.com/87540453/177857347-c06517eb-30b5-43be-862a-3e2a2918d42a.png)

Nele observamos que doenças cardiovasculares são mais comuns nos homens do que nas mulheres.

### Análise sobre o avanço da idade
Na imagem a seguir temos um gráfico de probabilidade relativa onde é mostrado que ao decorrer da idade a probabilidade de ter doenças cardio vasculares aumenta gradativamente

![relatividade_prob](https://user-images.githubusercontent.com/87540453/177858190-1702a87f-0517-4644-8884-c217d4d8221a.png)

Podemos relatar que, de acordo com os dados, a maior taxa de doenças presentes é entre pessoas de 60 à 64 anos.

### Relacionando com o IMC
Na imagem a seguir, temos um histograma que faz uma relação entre o IMC e o fato da pessoa ter ou não alguma doença cardiovascular.

![imc_graph](https://user-images.githubusercontent.com/87540453/177860902-4b79fccd-d442-4297-86a6-3b9262aa0ba7.png)


O processo para criar esse gráfico é muito simples, primeiros calculamos o IMC da pessoas presentes no nosso dataset e então dividimos em categorias.

Essas categórias são:
```
Menor que 18.5 - Abaixo do peso (Categoria 1)
Entre 18.5 e 24.9 - Peso dentro da normalidade (Categoria 2)
Entre 25 e 29.9 - Acima do Peso (Categoria 3)
Entre 30 e 34.9 - Obeso (Categoria 4)
Entre 35 e 39.9 - Obesidade Severa (Categoria 5)
Maior que 40 - Obesidade Morbida (Categoria 6)
```
Os Dados sobre as classificações foram retirados da [World Health Organization](https://www.who.int/europe/news-room/fact-sheets/item/a-healthy-lifestyle---who-recommendations).

É de fácil observação que conforme o aumento do IMC, aumenta o número de pacientes com doenças cardiovasculares.
