import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
import io
from get_chefkoch import Recipe, Search

model = load_model("veg_model.h5")

vegetable_dict = {
    0: "Bohne",
    1: "Bittergurke",
    2: "Flaschenkürbis",
    3: "Aubergine",
    4: "Brokkoli",
    5: "Kohl",
    6: "Paprika",
    7: "Karrote",
    8: "Blumenkohl",
    9: "Gurke",
    10: "Papaya",
    11: "Kartoffel",
    12: "Kürbis",
    13: "Radieschen",
    14: "Tomaten"
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
    
    # img_arr_2 shape -> (1, 100, 100, 3)
    # 1 -> batch size
    img_arr_2 = load_img(bytes_data)

    # array with probability for each class
    prediction = model.predict(img_arr_2)
    # st.write(prediction)
    
    # get the vegetable name
    category_name = vegetable_dict[np.argmax(prediction)]

    # get the probability    
    prob = np.max(prediction)
    rounded_prob = round(prob, 2)

    st.write(category_name + " - probability: " + str(rounded_prob))
    
    # get the recipe
    s = Search(category_name)
    recipe = s.recipes(limit=1)[0]
    st.subheader(recipe.name)

    # recipe image
    st.image(recipe.image)

    st.write("Dauer: " + str(recipe.totalTime) + " Minuten")

    st.write("Zutaten:")
    st.write(recipe.ingredients)


    st.write("Beschreibung:")
    st.write(recipe.description)