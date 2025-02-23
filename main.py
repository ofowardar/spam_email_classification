from scripts.model import modelling
from scripts.load_data import load_data
from scripts.preprocessing import preprocessText
import pandas as pd



######################################### LOAD DATA #######################################

data_path = "C:/Users/Ömer Faruk Özvardar/Desktop/E-Mail Spam Classification/data/spam_dataset.csv"
target_column = "message_content"

df = load_data(data_path=data_path)

######################################### DATA PREPROCESSING #######################################


df_preprocessed = preprocessText(df=df,target_column=target_column)


######################################### Modelling #######################################

model, vectorizer, acc = modelling(df_preprocessed)


######################################### TESTING #######################################


test_text = ["Congratulations, you've won a prize! Call us now to claim it. Your account has been selected for a special reward. Click here to claim it now! For more details, visit our website or contact us directly."]

test_text_vector = vectorizer.transform(test_text)

prediction = model.predict(test_text_vector)

print("Prediction:", prediction)