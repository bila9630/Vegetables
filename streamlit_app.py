import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
from get_chefkoch import Search

st.set_page_config(
    page_title="Vegetable ML",
    page_icon="🍱",
)

model_cnn = load_model("veg_model.h5")
model_tf = load_model("veg_tf.h5")

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


def get_prediction_cnn(img_bytes):
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
    
    # array with probability for each class
    prediction = model_cnn.predict(img_arr_2)

    # get the vegetable name
    category_name = vegetable_dict[np.argmax(prediction)]

    # get the probability
    prob = np.max(prediction)
    rounded_prob = round(prob, 2)

    return category_name, rounded_prob


def get_prediction_transfer_learning(img_bytes):
    # read image as bytes and resize to 224x224
    img_resized = image.load_img(io.BytesIO(img_bytes), target_size=(224, 224))

    # flatten img array
    img_arr = img_to_array(img_resized)

    # Add extra dimension to fit model
    img_arr_2 = np.expand_dims(img_arr, axis=0)

    # preprocess func for inceptionV3 from the library itself
    img_arr_3 = preprocess_input(img_arr_2)

    # array with probability for each class
    prediction = model_tf.predict(img_arr_3)

    # get the vegetable name
    category_name = vegetable_dict[np.argmax(prediction)]

    # get the probability
    prob = np.max(prediction)
    rounded_prob = round(prob, 2)

    return category_name, rounded_prob


st.title("Yummy vegetables!")
st.write("Simple upload a picture of a vegetable and we will generate a recipe for you!")

img_file_buffer = st.camera_input("Take a picture")

option = st.selectbox(
    "Select your model",
    ("CNN", "Transfer Learning (InceptionV3)"))

st.write('You selected:', option)


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

    if option == "CNN":
        category_name, rounded_prob = get_prediction_cnn(bytes_data)
    else:
        category_name, rounded_prob = get_prediction_transfer_learning(
            bytes_data)

    st.write(category_name + " - probability: " + str(rounded_prob))

    # get the recipe
    s_test = Search(category_name)

    # recipes is a list of Recipe urls
    recipes = s_test.recipes(limit=5)

    current_recipe_index = 0

    # Create two columns
    col1, col2 = st.columns(2)

    # Create buttons to show the next and previous recipe
    if col1.button("Previous Recipe"):
        current_recipe_index -= 1
        if current_recipe_index < 0:
            current_recipe_index = 0

    if col2.button("Next Recipe"):
        current_recipe_index += 1
        if current_recipe_index >= len(recipes):
            current_recipe_index = len(recipes) - 1

    display_recipe(recipes[current_recipe_index])
