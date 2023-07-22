# Vegetables_ML
Check out our application:

## Project description
Idea: Developing an application that identify vegetables
<br>The dataset is available on Kaggle: https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset
<br>Project: Vegetables ML
<br>Group: Hannah Schult, Sofie Pischl, Viet Duc Kieu

## Navigation
- **Model Development** can be found in the folder [/analytics](/analytics)
- **Data** can be found in the folder [/data](/data)

## application
### how to start locally
```
pip install -r requirements.txt
streamlit run streamlit_app.py
```
application is now running on http://localhost:8501

to freeze the requirements:
```
pip freeze > requirements.txt
```

to create a virtual environment:
```
# create a virtual environment
virtualenv env
# activate the virtual environment
env\Scripts\activate
```

## Trouble shooting
When the streamlit app is not running locally:
- adjust the path to the model in the streamlit_app.py file (there are comment that you just need to uncomment in the file)

## data
source: https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset
