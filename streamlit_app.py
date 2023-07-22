import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
import io
from get_chefkoch import Search

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
    # read image as bytes
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # resize image to 100x100
    img = img.resize((100, 100))

    # flatten img array
    img_arr = img_to_array(img)

    # reduce img values to 0-1
    img_scaled = img_arr / 255.0

    # expand dimension to fit model (1, 100, 100, 3)
    # 1 -> batch size
    img_arr_2 = np.expand_dims(img_scaled, axis=0)
    return img_arr_2


st.title("Yummy vegetables!")
st.write("Simple upload a picture of a vegetable and we will generate a recipe for you!")

img_file_buffer = st.camera_input("Take a picture")


def display_recipe(recipe):
    st.subheader(recipe.name)
    st.image(recipe.image)
    st.write("Dauer: " + str(recipe.totalTime) + " Minuten")
    st.write("Zutaten:")
    st.write(recipe.ingredients)
    st.write("Beschreibung:")
    st.write(recipe.description)


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
    s_test = Search(category_name)

    # recipes is a list of Recipe urls
    recipes = s_test.recipes(limit=5)

    current_recipe_index = 0

    # Create buttons to show the next and previous recipe
    col1, col2 = st.columns(2)

    if col1.button("Previous Recipe"):
        current_recipe_index -= 1
        if current_recipe_index < 0:
            current_recipe_index = 0

    if col2.button("Next Recipe"):
        current_recipe_index += 1
        if current_recipe_index >= len(recipes):
            current_recipe_index = len(recipes) - 1

    display_recipe(recipes[current_recipe_index])
