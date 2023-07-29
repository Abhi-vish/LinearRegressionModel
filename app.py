import streamlit as st    
import pandas as pd  
import matplotlib.pyplot as plt
import plotly 
from plotly import graph_objs as go
import time
import pickle
import os

data = pd.read_csv(os.path.join("model", "Boston.csv"))
data = data.drop(columns="Unnamed: 0")
data_tbl = data.head(n=20)

st.title("House Price Prediction")

nav = st.sidebar.selectbox("Navebar",["Home","Prediction","Contribute"],on_change=None)

if nav == "Home":
    image_path = os.path.join("static", "pexels-scott-webb-1029599.jpg")
    st.image(image_path)

    st.write("""
The Boston Housing Dataset is a famous and widely used dataset in machine learning and statistics. It was collected by researchers from the UCI Machine Learning Repository and is often used for regression analysis and predictive modeling tasks. The dataset contains information about housing in the Boston, Massachusetts area and was first published by Harrison and Rubinfeld in 1978.

The dataset consists of 14 attributes (features) and 1 target variable (response variable) as follows:

CRIM: Per capita crime rate by town.\n
ZN: Proportion of residential land zoned for lots over 25,000 sq. ft.\n
INDUS: Proportion of non-retail business acres per town.\n
CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise).\n
NOX: Nitric oxides concentration (parts per 10 million).\n
RM: Average number of rooms per dwelling.\n
AGE: Proportion of owner-occupied units built before 1940.\n
DIS: Weighted distances to five Boston employment centers.\n
RAD: Index of accessibility to radial highways.\n
TAX: Full-value property tax rate per $10,000.\n
PTRATIO: Pupil-teacher ratio by town.\n
B: 1000(Bk - 0.63)^2 where Bk is the proportion of Black people by town.\n
LSTAT: % lower status of the population.\n
MEDV: Median value of owner-occupied homes in $1000s (target variable).
""")
    st.markdown("<br>", unsafe_allow_html=True)

    st.header("Visualize the Dataset")
    
    if st.checkbox("Show DataFrame Table"):
        # progress = st.progress(0)
        # for i in range(100):
        #     time.sleep(0.05)
        #     progress.progress(i+1)
        st.table(data_tbl)
    
    graph = st.selectbox("# What kind of graph ?",['Non-Interactive','Interactive'],True)

    if graph == "Non-Interactive":
        x = data.drop(columns='medv')
        x_features = x.columns
        x_axis = st.selectbox("Select feature",x_features)
        fig, ax = plt.subplots()
        plt.xlabel(x_axis)
        plt.ylabel('medv')
        plt.scatter(data[x_axis],data['medv'])
        st.pyplot(fig)
    elif graph == "Interactive":
        x = data.drop(columns='medv')
        x_features = x.columns
        x_axis = st.selectbox("Select feature", x_features)

        layout = go.Layout(
            xaxis=dict(range=[0, 80]),
            yaxis=dict(range=[0, 60])
        )

        fig = go.Figure(data=go.Scatter(x=data[x_axis], y=data['medv'], mode='markers'))
        st.plotly_chart(fig, layout=layout)

elif nav == "Prediction":

    with open('model\linearmodel.pkl', 'rb') as file:
        model = pickle.load(file)

    st.header("Predict House Price")

# crim	zn	indus	chas	nox	rm	age	dis	rad	tax	ptratio	black	lstat	
    crim,zn = st.columns(2)
    cr = crim.number_input(label="crim",step=1.,format="%.3f")
    zn= zn.number_input("zn",step=1.,format="%.3f")

    indius,chas = st.columns(2)
    ind = indius.number_input("indus",step=1.,format="%.3f")
    cha = chas.number_input("chas",step=1.,format="%.3f")

    nox,rm = st.columns(2)
    nox = nox.number_input("nox",step=1.,format="%.3f")
    rm = rm.number_input("rm",step=1.,format="%.3f")

    age,dis = st.columns(2)
    age = age.number_input('age',step=1.,format="%.3f")
    dis = dis.number_input("dis",step=1.,format="%.3f")

    rad,tax = st.columns(2)
    rad = rad.number_input('rad',step=1.,format="%.3f")
    tax = tax.number_input("tax",step=1.,format="%.3f")
    

    ptr,bla = st.columns(2)
    ptrratio = ptr.number_input("ptratio",step=1.,format="%.3f")
    black = bla.number_input("black",step=1.,format="%.3f")

    lstat = st.number_input("lstat",step=1.,format="%.3f")
    # for col in data:
    #     st.write(min(data[col]),max(data[col]))

    data = [cr,zn,ind,cha,nox,rm,age,dis,rad,tax,ptrratio,black,lstat]

    if st.button('predict'):

        prediction = model.predict([data])
        st.success(prediction* 1000)


elif nav == "Contribute":
    st.write("Contirbute")


    # 0.0063	18.0000	2.3100	0	0.5380	6.5750	65.2000	4.0900	1	296	15.3000	396.9000	4.9800	24.0000
