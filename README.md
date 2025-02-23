# 📧 E-Mail Spam Classification

## 🔍 Project Overview

The **E-Mail Spam Classification** project aims to develop a machine learning model capable of classifying emails into two categories: **Spam** and **Not Spam**. The dataset used in this project contains a variety of emails, which allows the model to learn from diverse features. The process includes data preprocessing, feature extraction, model training, evaluation, and testing with new input data.

## ⚙️ Features

- **Preprocessing**: Cleaning and preparing email text data by removing unnecessary information such as stop words, punctuations, numerical values, and performing lemmatization.
- **Vectorization**: Converting text data into numerical features using **TF-IDF** (Term Frequency-Inverse Document Frequency) to make it suitable for machine learning algorithms.
- **Modeling**: Training a **Logistic Regression** model to classify emails based on preprocessed content.
- **Evaluation**: Evaluating the model using performance metrics such as **accuracy**, **precision**, **recall**, and **F1-score**.
- **Testing**: Predicting the classification of new emails based on the trained model.

## 📥 Installation

### 🔑 Prerequisites

Ensure you have **Python 3.x** installed on your machine.

You also need to install the following Python libraries:

- `pandas`
- `numpy`
- `scikit-learn`
- `nltk`
- `time`

📂 File Structure

```
.
├── data/
│   └── spam_dataset.csv        # Dataset containing email data
├── scripts/
│   ├── load_data.py            # Loads the data
│   ├── preprocessing.py        # Handles data preprocessing
│   ├── model.py                # Builds, trains, and evaluates the model
├── main.py                     # Main script that ties everything together
└── README.md                   # Project documentation
```

🏗️ How to Use
1. Load the Data
The dataset (spam_dataset.csv) is loaded using the load_data function in the scripts/load_data.py file.

2. Preprocess the Data
Before training the model, the email text data needs to be preprocessed. The preprocessText function in scripts/preprocessing.py is responsible for:

Removing stop words
Removing punctuation
Normalizing text (lowercasing)
Removing numerical values
Lemmatizing words

3. Train the Model
The modelling function in scripts/model.py handles the training process:

It splits the data into training and test sets.
It uses TF-IDF vectorization to convert text into numeric features.
It trains a Logistic Regression model to classify the emails as spam or not spam.
4. Test the Model
After training, you can test the model using the following code:


test_text = ["Hello Lonnie Just wanted to touch base regarding our project’s next steps. Please find the details below..."]
prediction = model.predict(vectorizer.transform(test_text))
print(f"Prediction: {prediction}")
This will output the model's prediction for whether the provided email is spam or not.

📊 Model Evaluation
The model's performance is evaluated using various metrics such as:

Accuracy
Precision
Recall
F1-Score
These metrics are printed out during the model training process.

🤝 Contributions
Feel free to fork this repository and submit pull requests. All contributions are welcome, whether it’s code improvements, bug fixes, or new features. 😊

📜 License
This project is licensed under the MIT License.
