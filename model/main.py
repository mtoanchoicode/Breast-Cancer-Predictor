import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle as pickle

def get_clean_data():
    data = pd.read_csv("data/data.csv")
    
    #Drop the last column because not using it (axis 1 = drop column)
    data = data.drop(['Unnamed: 32', 'id'], axis = 1)

    #encode the diagnoses
    data['diagnosis'] = data['diagnosis'].map({'M': 1, "B": 0})

    return data

def create_model(data):
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']

    #scale data to same scale to ensure uniform
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    #split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=14)

    #training test
    model = LogisticRegression()
    model.fit(X_train, y_train)

    #test model
    y_predict = model.predict(X_test)
    print('Accuracy of our model: ', accuracy_score(y_test, y_predict))
    print('Classification report: \n', classification_report(y_test, y_predict))

    return model, scaler


def main():
    data = get_clean_data()
    
    model, scaler = create_model(data)

    with open('model/model.pkl', 'wb') as file:
        pickle.dump(model, file)
    with open('model/scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)






#Just execute when calling not from importing
if __name__ == '__main__':
    main()