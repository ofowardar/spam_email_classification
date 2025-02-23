from scripts.load_data import load_data
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import time
# nltk.download("stopwords")
# nltk.download("punkt")
# nltk.download("wordnet")

############################################################################################################################################################


def preprocessText(df, target_column):
    try:
        print("Copying the Database...")
        df_preprocessing = df.copy()

        df_preprocessing[target_column] = df_preprocessing[target_column].astype(str).fillna("")

        print("Removing Stop Words...")
        stopws = set(stopwords.words('english'))
        df_preprocessing[target_column] = df_preprocessing[target_column].apply(
            lambda x: " ".join(word for word in x.split() if word.lower() not in stopws)
        )
        time.sleep(2)

        print("Removing Punctuations...")
        df_preprocessing[target_column] = df_preprocessing[target_column].replace(r'[^\w\s]', '', regex=True)
        time.sleep(2)

        print("Normalizing Case...")
        df_preprocessing[target_column] = df_preprocessing[target_column].apply(
            lambda x: ' '.join(word.lower() for word in x.split())
        )
        time.sleep(2)

        print("Removing Numerical Values...")
        df_preprocessing[target_column] = df_preprocessing[target_column].replace(r'\d+', '', regex=True)
        time.sleep(2)

        print("Lemmatizing Words...")
        lemmatizer = WordNetLemmatizer()
        df_preprocessing[target_column] = df_preprocessing[target_column].apply(
            lambda x: " ".join(lemmatizer.lemmatize(word) for word in x.split())
        )

        print(df_preprocessing.head(30))

        return df_preprocessing

    except LookupError as e:
        print(f"NLTK dataset is missing: {e}. Please run the 'nltk.download()' commands.")
        return None

    except AttributeError as e:
        print(f"A data type error occurred: {e}. Check for missing values and use `astype(str)`.") 
        return None

    except Exception as e:
        print(f"An unknown error occurred: {e}")
        return None

    finally:
        print("Preprocessing function has ended.")



