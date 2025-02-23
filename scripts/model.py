from scripts.load_data import load_data
from scripts.preprocessing import preprocessText
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score,roc_auc_score
import time


def modelling(df):

    X = df["message_content"]
    y = df["is_spam"]

    vectorizer = TfidfVectorizer()

    X_vector = vectorizer.fit_transform(X)
    print(X.shape,y.shape)
    X_train,X_test,y_train,y_test = train_test_split(X_vector,y,random_state=42,test_size=0.2)

    try:
        print("Logistic Regression creating...")
        time.sleep(2)
        model = LogisticRegression()
        print("Training Model...")
        time.sleep(2)
        model.fit(X_train,y_train)
        print("Testing For Metrics...")
        model_predict = model.predict(X_test)
    except:
        print("Something Went Wrong...")

    # Metrics
    print(classification_report(y_test,model_predict))
    accuracy = accuracy_score(y_test,y_pred=model_predict)
    print(accuracy)

    return model, vectorizer, accuracy









