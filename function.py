import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

def clean_data(df, df2):  

    df2.drop(45, inplace=True)

    for col in df:
        df[col] = df[col].str.strip()


    df.fillna(0, inplace=True)

    data1 = np.array(df.iloc[:, 1:18])
    data2 = np.array(df2.iloc[:, 0])
    weight = np.array(df2.iloc[:, 1])

    for i in range(data1.shape[0]):
        for j in range(data1.shape[1]):
            for k in range(data2.size):
                if data2[k] == data1[i][j]:
                    data1[i][j] = weight[k]
                    break

    # columns_name = df.columns.unique()

    temp = pd.DataFrame(data1, columns=['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4',
        'Symptom_5', 'Symptom_6', 'Symptom_7', 'Symptom_8', 'Symptom_9',
        'Symptom_10', 'Symptom_11', 'Symptom_12', 'Symptom_13', 'Symptom_14',
        'Symptom_15', 'Symptom_16', 'Symptom_17'])
    temp = df.iloc[:, [0]].join(temp)

    nonNums = []

    for index, row in temp.iterrows():
        if type(row['Symptom_1']) != np.int64:
            nonNums.append(row['Symptom_1'])
        elif type(row['Symptom_2']) != np.int64:
            nonNums.append(row['Symptom_2'])
        elif type(row['Symptom_3']) != np.int64:
            nonNums.append(row['Symptom_3'])
        elif type(row['Symptom_4']) != np.int64:
            nonNums.append(row['Symptom_4'])
        elif type(row['Symptom_5']) != np.int64:
            nonNums.append(row['Symptom_5'])
        elif type(row['Symptom_6']) != np.int64:
            nonNums.append(row['Symptom_6'])
        elif type(row['Symptom_7']) != np.int64:
            nonNums.append(row['Symptom_7'])
        elif type(row['Symptom_8']) != np.int64:
            nonNums.append(row['Symptom_8'])
        elif type(row['Symptom_9']) != np.int64:
            nonNums.append(row['Symptom_9'])
        elif type(row['Symptom_10']) != np.int64:
            nonNums.append(row['Symptom_10'])
        elif type(row['Symptom_11']) != np.int64:
            nonNums.append(row['Symptom_11'])
        elif type(row['Symptom_12']) != np.int64:
            nonNums.append(row['Symptom_12'])
        elif type(row['Symptom_13']) != np.int64:
            nonNums.append(row['Symptom_13'])
        elif type(row['Symptom_14']) != np.int64:
            nonNums.append(row['Symptom_14'])
        elif type(row['Symptom_15']) != np.int64:
            nonNums.append(row['Symptom_15'])
        elif type(row['Symptom_16']) != np.int64:
            nonNums.append(row['Symptom_16'])
        elif type(row['Symptom_17']) != np.int64:
            nonNums.append(row['Symptom_17'])

    nonNums = np.unique(nonNums)

    nonNums = np.delete(nonNums, 0)

    indexs = []

    for index, row in df2.iterrows():
        if row['Symptom'] == 'dischromic_patches' :
            indexs.append(row['weight'])
        elif row['Symptom'] == 'foul_smell_ofurine':
            indexs.append(row['weight'])
        elif row['Symptom'] == 'spotting_urination':
            indexs.append(row['weight'])

    temp = temp.replace('dischromic _patches', indexs[0])
    temp = temp.replace('foul_smell_of urine', indexs[1])
    df = temp.replace('spotting_ urination', indexs[2])
    return df

def split_data(df, n, test_s, random_s):
    X = df.iloc[:, 1:(n+1)]
    y = df.iloc[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_s, random_state=random_s)
    return X_train, X_test, y_train, y_test

def convert_to_weight(symptom_temp, df):
    for index, row in df.iterrows():
        for i in range(len(symptom_temp)):
            if row['Symptom'] == symptom_temp[i]:
                symptom_temp[i] = row['weight']
                break
    return symptom_temp

def add_length(symptom_temp, n):
    if len(symptom_temp) < n:
        symptom_temp.extend([0] * (n - len(symptom_temp)))
    return symptom_temp

def train_data(X_train, y_train, classifier_name, parameters):
    if classifier_name == "KNN":
        model = KNeighborsClassifier(n_neighbors=parameters['KNN'])
        model.fit(X_train, y_train)
    elif classifier_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=parameters['n_estimators'], max_depth=parameters['max_depth'])
        model.fit(X_train, y_train)
    return model

def predict_disease(model, symptom_temp):
    disease = model.predict([symptom_temp])
    return disease

def disease_description(disease, df):
    for index, row in df.iterrows():
        if row['Disease'] == disease:
            return row['Description']

def search_precaution(disease, df):
    temp_precaution = []

    for index, row in df.iterrows():
        if row['Disease'] == disease:
            for i in range(len(row)):
                if i == 0:
                    continue

                if row[i] != '':
                    temp_precaution.append(row[i])
    
    return temp_precaution

def count_accuracy(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

