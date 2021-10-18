import streamlit as st
import pickle
import pandas as pd
import numpy as np
from PIL import Image


def rent_pred():
    st.subheader('Seller Type')
    seller = rent['seller_type'].unique()
    seller_type = st.radio('', seller)

    st.subheader('No. of Bedrooms')
    bedroom = st.slider('', 1, 6)

    st.subheader('Layout Type')
    layout = rent['layout_type'].unique()
    layout_type = st.radio('', layout)

    st.subheader('Property Type')
    prop = rent['property_type'].unique()
    property_type = st.selectbox('', prop)

    st.subheader('Localities')
    localities = rent['locality'].unique()
    locality = st.selectbox('', localities)

    st.subheader('Area')
    area = st.slider('', 100, 7000)

    st.subheader('Furnishing')
    furnish = rent['furnish_type'].unique()
    furnish_type = st.radio('', furnish)

    st.subheader('No. of Bathrooms')
    bathroom = st.slider('', 1, 9)

    rent_pred_data = {
        'bedroom': bedroom,
        'area': area,
        'bathroom': bathroom,
    }

    df2 = pd.DataFrame(rent_pred_data, index=[0])

    return df2


model = pickle.load(open('Mumbai_rent_model.sav', 'rb'))
rent = pd.read_csv("Clean_Mumbai_rent.csv")
st.title("Mumbai House Rent Prediction")
st.subheader("This model will help you predict rent around different localities of Mumbai.")
image = Image.open("header-img.jpg")
st.image(image, '')

pred_data = rent_pred()
price = model.predict(pred_data)
st.title("PREDICTED RENT")
st.header(str('Rs. ' + str(np.round(price[0], 2))))


