import pandas as pd 
import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


# função para carregar o Dataset
@st.cache
def get_data():
    return pd.read_csv('Dados/train.csv')

# função para treinar o modelo
def train_model():
    data = get_data()
    x_train = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    x_train['Age'] = x_train['Age'].fillna(x_train['Age'].median())

    x_train = x_train.dropna(axis=0)

    target = x_train['Survived']

    x_train = x_train.drop(['Survived'], axis=1)

    x_train['Sex'] = x_train[['Sex']].apply(LabelEncoder().fit_transform)
    x_train['Embarked'] = x_train[['Embarked']].apply(LabelEncoder().fit_transform)

    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    rf_model.fit(x_train, target)

    return rf_model


# criando um dataframe
data = get_data()

# título
st.title("Você Sobreviveria ao Naufrágio do Titanic?")

# subtítulo
st.markdown("Este é um Data App utilizado para exibir a solução de Machine Learning para o problema de predição de sobreviventes ao naufráfio do Titanic.")


st.image('https://miro.medium.com/max/480/1*bPL6FfFOkz-FUPsABMR31Q.gif',width=600)

st.subheader('Resultado da predição:')

# Subtitulo
st.sidebar.subheader("Defina os atributos do passageiro para predição")

# Imput de informacoes

name = st.sidebar.text_input("Infome seu nome")
sex = st.sidebar.selectbox('Escolha seu sexo', data['Sex'].unique())
# Transformando em dados binarios
sex = 1 if sex == 'male' else 0
age = st.sidebar.number_input("Informe sua idade",value=data['Age'].mean())
sibsp= st.sidebar.number_input('Informe a quantidade de cônjuges e irmãos a bordo',value=data['SibSp'].mean())
parch = st.sidebar.number_input('Informe quantidade de pais e filhos a bordo',value=data['Parch'].mean())
points_embarked = ['Cherbourg', 'Queenstown', 'Southampton']
embarked = st.sidebar.selectbox('Escolha o local de embarque', points_embarked)

if embarked == "Cherbourg":
    e_s = "C"
    e_d = 0
elif embarked == "Queenstown":
    e_s = 'Q'
    e_d = 1
else:
    e_s = 'S'
    e_d = 2

l_class = ['1ª Classe', '2ª Classe', '3ª Classe']
pclass = st.sidebar.selectbox('Escolha a classe para viagem',l_class )
if pclass == '1ª Classe':
    pclass_n = 1
elif pclass == '2ª Classe':
    pclass_n = 2
else:
    pclass_n = 3

if pclass:
    data_filter = data[((data['Embarked'] == e_s) & (data['Pclass'] == pclass_n))]
    fare = round(data_filter['Fare'].mean(), 2)
    st.sidebar.markdown('Valor do ticket: $ {}'.format(fare))
test = [[pclass_n, sex, age, sibsp, parch, fare, e_d]]
model = train_model()
ver = st.sidebar.button('Realizar predição?')
if ver:
    predict = model.predict(test)
    if predict == 0:
        st.markdown("{}, infelizmente você não sobreviveria".format(name))
    else:
        st.markdown("{}, felizmente você sobreviveria".format(name))



