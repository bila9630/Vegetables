import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
import io

model = load_model("veg_model.h5")

vegetable_dict = {
    0: 'Bean',
    1: 'Bitter_Gourd',
    2: 'Bottle_Gourd',
    3: 'Brinjal',
    4: 'Broccoli',
    5: 'Cabbage',
    6: 'Capsicum',
    7: 'Carrot',
    8: 'Cauliflower',
    9: 'Cucumber',
    10: 'Papaya',
    11: 'Potato',
    12: 'Pumpkin',
    13: 'Radish',
    14: 'Tomato'
}


def load_img(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((100, 100))
    img_arr = img_to_array(img)
    img_scaled = img_arr / 255.0
    img_arr_2 = np.expand_dims(img_scaled, axis=0)
    return img_arr_2


st.title("Yummy vegetables!")
st.write("Simple upload a picture of a vegetable and we will generate a recipe for you!")

img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    bytes_data = img_file_buffer.getvalue()
    img_arr_2 = load_img(bytes_data)
    st.write(img_arr_2.shape)
    prediction = model.predict(img_arr_2)
    st.write(prediction)
    
    # get the vegetable name
    category_name = vegetable_dict[np.argmax(prediction)]
    st.write(category_name)

    # get the probability    
    prob = np.max(prediction)
    rounded_prob = round(prob, 2)
    st.write(rounded_prob)

    st.write(category_name + " - probability: " + str(rounded_prob))
    
    