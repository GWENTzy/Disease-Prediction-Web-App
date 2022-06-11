import streamlit as st 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px

from mpl_toolkits.mplot3d import Axes3D
from sklearn import tree
from PIL import Image
from sklearn import preprocessing
from function import *

def home():
    st.markdown("<h2 style='text-align: center; color: white;'>Disease Prediction</h2>", unsafe_allow_html=True)

    st.markdown('<br />', unsafe_allow_html=True)

    df = pd.read_csv('data/dataset.csv')
    df2 = pd.read_csv('data/symptom-severity.csv')
    df3 = pd.read_csv('data/symptom_precaution.csv')
    df4 = pd.read_csv('data/symptom_Description.csv')

    final_df = clean_data(df, df2)
    label_encoder = preprocessing.LabelEncoder()
    c = label_encoder.fit_transform(final_df['Disease'])
    total = 0

    st.write('Select independent variables')
    symptom_1 = st.checkbox('Symptom_1')
    if symptom_1:
        total += 1
    symptom_2 = st.checkbox('Symptom_2')
    if symptom_2:
        total += 1
    symptom_3 = st.checkbox('Symptom_3')
    if symptom_3:
        total += 1
    symptom_4 = st.checkbox('Symptom_4')
    if symptom_4:
        total += 1
    symptom_5 = st.checkbox('Symptom_5')
    if symptom_5:
        total += 1
    symptom_6 = st.checkbox('Symptom_6')
    if symptom_6:
        total += 1

    # X_train, X_test, y_train, y_test = split_data(final_df, total)

    def add_parameter(classifier_name):
        parameters = dict()
        if classifier_name == "KNN":
            parameters['KNN'] = st.slider('K', 1, 15)
        else:
            max_depth = st.slider('Max Depth', 2, 15)
            parameters['max_depth'] = max_depth
            n_estimators = st.slider('N Estimators', 1, 100)
            parameters['n_estimators'] = n_estimators
        return parameters

    isFilled = False

    col1, col2 = st.columns(2)
    with col1:

        # st.write("""
        # # Explore different classifier and datasets
        # Which one is the best?
        # """)

        # st.sidebar.markdown("<h1 style='text-align: center; color: white;'>Classification Algorithm</h1>", unsafe_allow_html=True)
        classifier_name = st.selectbox(
            'Select classifier',
            ('KNN', 'Random Forest')
        )

        parameters = add_parameter(classifier_name)

        if classifier_name == "KNN":
            st.success(f'You have selected {classifier_name} with ' + str(parameters['KNN']) + ' Nearest Neighbours')
        else:
            st.success(f'You have selected {classifier_name} with ' + str(parameters['max_depth']) + ' Max Depth and ' + str(parameters['n_estimators']) + ' N Estimators')

    with col2:
        test_s = st.slider('Test size', 0.1, 0.9)
        random_s = st.slider('Random state', 0, 50)
        X_train, X_test, y_train, y_test = split_data(final_df, total, test_s, random_s)

    symptoms = df2['Symptom'].tolist()
    symptom_temp = st.multiselect('Choose your symptoms (Max. 6)', symptoms)

    if st.button('Submit'):
        if len(symptom_temp) != 0:
            if total == 0:
                st.warning('Please select atleast one independent variable')
            elif len(symptom_temp) > 6:
                st.warning('Please insert 6 symptoms only')
            else:
                isFilled = True
            
        else:
            st.warning('Please insert your symptom')

    st.markdown('<br />', unsafe_allow_html=True)

    if isFilled == True:
        model = train_data(X_train, y_train, classifier_name, parameters)
        acc = count_accuracy(model, X_test, y_test)
        st.write(f'Accuracy: {round((acc * 100), 2)}%')

    if(isFilled == True):
        symptom_temp = convert_to_weight(symptom_temp, df2)
        symptom_temp = add_length(symptom_temp, total)
        disease = predict_disease(model, symptom_temp)
        st.write(disease[0])

        with st.expander("View Description"):
            description = disease_description(disease, df4)
            st.write(description)

        with st.expander("View Precautions"):
            count = 1
            precautions = search_precaution(disease, df3)
            precautions = [precaution for precaution in precautions if str(precaution) != 'nan']
            for precaution in precautions:
                st.write(str(count) + '. ' + precaution)
                count += 1
        
    st.markdown('<br />', unsafe_allow_html=True)

    if(isFilled == True):
        st.markdown("<h2 style='text-align: center; color: white;'>Chart</h2>", unsafe_allow_html=True)
        if classifier_name == "KNN":
            with st.container():

                z = final_df['Symptom_4']
                x = final_df['Symptom_1']
                y = final_df['Symptom_2']
                
                # # Creating figure
                # fig = plt.figure(figsize = (16, 9))
                # ax = plt.axes(projection ="3d")
                
                # # Add x, y gridlines
                # ax.grid(b = True, color ='grey',
                #         linestyle ='-.', linewidth = 0.3,
                #         alpha = 0.2)
                
                
                # # Creating color map
                # my_cmap = plt.get_cmap('hsv')
                
                # # Creating plot
                # sctt = ax.scatter3D(x, y, z,
                #                     alpha = 0.8,
                #                     c = c,
                #                     cmap = my_cmap)
                
                # ax.set_xlabel('Symptom 1', fontweight ='bold')
                # ax.set_ylabel('Symptom 2', fontweight ='bold')
                # ax.set_zlabel('Symptom 4', fontweight ='bold')
                # fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5)

                # legend
                # plt.legend(*sctt.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)
                
                fig = px.scatter_3d(final_df, x=x, y=y, z=z, color=final_df['Disease'], symbol=final_df['Disease'], size=final_df['Symptom_3'], width=800, height=800)

                # show plot
                st.plotly_chart(fig, use_container_width=True)
        else:
            with st.container():
                fig, ax = plt.subplots(figsize=(25, 15))
                ax = tree.plot_tree(model.estimators_[0], feature_names=X_train.columns, class_names=y_train.unique() , filled=True, fontsize=14, impurity=True, rounded=True)
                st.pyplot(fig)