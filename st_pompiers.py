#IMPORT BIBLIOTHEQUES
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import sklearn.metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV


#IMPORT CSV
df = pd.read_csv("df_pompierV1.csv")


#CREATION DES PAGES
st.sidebar.title("Sommaire")
pages = ["Contexte du projet", "Exploration des données", "Modélisation", "Conclusion"]
page = st.sidebar.radio("Aller vers la page:", pages)


#CREDITS
st.sidebar.title("Crédits")
st.sidebar.markdown("""
<style>
.sidebar-box {
    background-color: #E6E6E6;
    padding: 10px;
    border-radius: 5px;
    margin-top: 10px;
}
.sidebar-box strong {
    font-weight: bold;
}
.sidebar-box a {
    color: #1f77b4;
    text-decoration: none;
}
</style>
<div class="sidebar-box">
<strong>Auteurs :</strong><br>
- Anaïs CARCANADE <a href="https://www.linkedin.com/in/ana%C3%AFs-carcanade-828948aa">linkedIn</a><br>
- Michaël DAMON<br><br>
<strong>Cursus :</strong><br>
Continu - Data Analyst - Jan 24<br>
<a href="https://www.datascientest.com">DataScientest</a><br>
<br>
<strong>Sources :</strong><br>
data.london.gov<br>
<a href="https://data.london.gov.uk/dataset/london-fire-brigade-incident-records">LFB Incident Records</a><br> <a href="https://data.london.gov.uk/dataset/london-fire-brigade-mobilisation-records">LFB Mobilisation Records</a>
</div>
""", unsafe_allow_html=True)


#CONTENU DES PAGES

#PAGE CONTEXTE DU PROJET
if page == pages[0] :
    st.write("""
    <style>
    .big-font {
        font-size: 50px !important;
    }
    </style>
    <p class="big-font">PyFire 🚒</p>
    """, unsafe_allow_html=True)
    st.write("### La Brigade des Pompiers de Londres")
    st.write("L’objectif de ce projet est d’analyser et de prédire les temps de réponse et de mobilisation de la Brigade des Pompiers de Londres.")
    st.write("Ce projet est réalisé dans le cadre de notre formation avec DataScientest.")
    st.image("image1.jpg")
    st.write("### Introduction")
    st.write("Pour mener à bien ce projet, deux datasets nous ont été fournis. Les incidents et les mobilisations.")
    st.write("Le premier jeu de données fourni contient les détails de chaque incident traité. Des informations sont fournies sur la date et le lieu de l'incident ainsi que sur le type d'incident traité. Le second jeu de données contient les détails de chaque camion de pompiers envoyé sur les lieux d'un incident. Des informations sont fournies sur l'appareil mobilisé, son lieu de déploiement et les heures d'arrivée sur les lieux de l'incident.")
    st.write("Les données s'étendent sur une période de 2018 à 2023 et concernent la ville de Londres.")
    st.image("image2.jpg")

#PAGE EXPLORATION DES DONNEES
elif page == pages[1]:
    st.write("### Pre-processing")
    st.write("Les datasets étaient relativement propres, nous avons tout de même du réaliser plusieurs étapes de pré-processing avant de pouvoir les exploiter pleinement. Comme par exemple :")
    st.button("valeurs manquantes")
    if st.button("Conversion"):
        code_date = '''df_pompier["DateOfCall"] = pd.to_datetime(df_pompier["DateOfCall"], dayfirst = True, errors='coerce')'''
        st.code(code_date, language='python')
    else:
        code_NA = '''df_pompier["DeployedFromStation_Name"] = df_pompier["DeployedFromStation_Name"].fillna(df_pompier["DeployedFromStation_Name"].mode()[0])'''
        st.code(code_NA, language='python')
    st.write("Après avoir nettoyé l'ensemble des variables concernées, nous passons directement à la présentation du dataframe final.")

    st.write("### Affichage du dataframe nettoyé")
    st.dataframe(df.head())
    st.write("Dimensions du dataframe:")
    st.write(df.shape)

    st.write("### Dataviz")
    st.write("Dans cette partie, nous projetons quelques graphiques qui nous ont parus pertinents lors de l'exploration des données.")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df["Groupe_incident"], bins = 3)
    plt.title("Répartition des groupes d'incident")
    st.pyplot(fig)
    st.write("On constate que les 3 catégories d'incidents ne sont pas très équilibrées, on a affaire majoritairement à des fausses alarmes.")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x = "Groupe_lieu", hue = "Groupe_incident", data = df)
    plt.xticks(rotation=90)
    plt.title("Répartition des groupes d'incident par environnement")
    st.pyplot(fig)
    st.write("On remarque que la majorité des incidents auxquels sont confrontés les pompiers de Londres surviennent dans les foyers - Dwelling. Au sein de ces logements, les fausses alarmes occupent une place prépondérante dans les types d'incidents survenus, suivis par les services spéciaux et les incendies.")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df["Duree_trajet"], bins = 50)
    plt.title("Distribution de la durée de trajet")
    st.pyplot(fig)
    Minimum = st.checkbox("Afficher le temps minimal")
    if Minimum:
        st.write(df['Duree_trajet'].min()/60)
    Moyenne = st.checkbox("Afficher le temps moyen")
    if Moyenne:
        st.write(round(df['Duree_trajet'].mean() / 60, 1))
    Maximum = st.checkbox("Afficher le temps maximal")
    if Maximum:
        st.write(df['Duree_trajet'].max()/60)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_hour = df['Duree_intervention'].groupby(df['HourOfCall']).mean()
    sns.barplot(x=bar_hour.index,y=bar_hour.values)
    plt.title("Le délais d'arrivée en fonction de l'heure de la journée")
    st.pyplot(fig)
    st.write("On constate des variations selon les heures, avec un délais plus court sur la matinée entre 7 et 9 h et la nuit après 21h")

#PAGE MODELISATION
elif page ==pages[2]:
    st.write("### La cible")
    st.write("La variable cible que nous avons retenue est la durée du trajet")
    code_cible = '''
    feats = df_pompier.drop('Duree_trajet' , axis = 1 )
    target = df_pompier['Duree_trajet']

    X_train , X_test , y_train , y_test = train_test_split(feats , target , test_size = 0.2 , random_state = 42 )'''
    st.code(code_cible, language='python')

    st.write("### Modélisation")
    df_pompier = pd.read_csv("df_pompierV1.csv")

    columns_to_drop2 = ['Incident_ID','Mobilisation_ID','Nb_appels','DateAndTimeMobilised','DateAndTimeArrived','DateAndTimeOfCall','DateOfCall','TimeOfCall','YearOfCall','Delay_motif','Duree_intervention']
    df_pompier = df_pompier.drop(columns=columns_to_drop2)
    
    feats = df_pompier.drop('Duree_trajet' , axis = 1 )
    target = df_pompier['Duree_trajet']

    X_train , X_test , y_train , y_test = train_test_split(feats , target , test_size = 0.2 , random_state = 42 )

    #Encodage
    columns_to_encode = [
    'Caserne_mob', 'Caserne_inc', 'Groupe_incident', 'Type_incident',
    'Groupe_lieu', 'Type_lieu', 'Adress_postcode', 'Adress_borough', 'Adress_ward']
    encoder = ce.OrdinalEncoder(cols=columns_to_encode, handle_unknown='impute')
    X_train = encoder.fit_transform(X_train)
    X_test = encoder.transform(X_test)

    #Standardisation
    numeric_columns = ['HourOfCall','Caserne_mob','Caserne_inc', 'Delay_ID',
                    'Groupe_incident','Type_incident','Groupe_lieu','Type_lieu','Adress_postcode',
                    'Adress_borough','Adress_ward','Easting_rounded','Northing_rounded']
    scaler = StandardScaler()
    X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
    X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])

    #Remplacement des vides
    X_test = X_test.fillna(0)

    #Joblib
    regressor_linear = joblib.load("modele_regressor_linear")
    regressor_tree = joblib.load("modele_regressor_tree")
    regressor_forest = joblib.load("modele_regressor_forest")
    regressor_gradient = joblib.load("modele_regressor_gradient")
    metrique = joblib.load("comparaison_metrique")
    hyperparametre = joblib.load("hyperparametre_Forest")

    #Prédictions
    y_pred_linear = regressor_linear.predict(X_test)
    y_pred_tree = regressor_tree.predict(X_test)
    y_pred_forest = regressor_forest.predict(X_test)
    y_pred_gradient = regressor_gradient.predict(X_test)

    #Choix du modèle
    modele_choisi = st.selectbox(label = "Modèle", options = ["Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting"])

    def train_model(modele_choisi):
        if modele_choisi == "Linear Regression":
            y_pred = y_pred_linear
        elif modele_choisi == "Decision Tree":
            y_pred = y_pred_tree
        elif modele_choisi == "Random Forest":
            y_pred = y_pred_forest
        elif modele_choisi == "Gradient Boosting":
            y_pred = y_pred_gradient
        r2 = r2_score(y_test, y_pred)
        return r2
    
    def graph_model(modele_choisi):
        if modele_choisi == "Linear Regression":
            feat_importances_linear = pd.DataFrame(regressor_linear.coef_, index=feats.columns, columns=["importance"])
            feat_importances_linear.sort_values(by="importance", ascending=False, inplace=True)
            ax = feat_importances_linear.plot(kind="bar", figsize=(8, 6))
            plt.title("Feature Importances - Linear Regression")
            plt.xlabel("Features")
            plt.ylabel("Importance")
            return ax.get_figure()
        elif modele_choisi == "Decision Tree":
            feat_importances_tree = pd.DataFrame(regressor_tree.feature_importances_, index=feats.columns, columns=["importance"])
            feat_importances_tree.sort_values(by="importance", ascending=False, inplace=True)
            ax = feat_importances_tree.plot(kind="bar", figsize=(8, 6))
            plt.title("Feature Importances - Decision Tree")
            plt.xlabel("Features")
            plt.ylabel("Importance")
            return ax.get_figure()
        elif modele_choisi == "Random Forest":
            feat_importances_forest = pd.DataFrame(regressor_forest.feature_importances_, index=feats.columns, columns=["importance"])
            feat_importances_forest.sort_values(by="importance", ascending=False, inplace=True)
            ax = feat_importances_forest.plot(kind="bar", figsize=(8, 6))
            plt.title("Feature Importances - Random Forest")
            plt.xlabel("Features")
            plt.ylabel("Importance")
            return ax.get_figure()
        elif modele_choisi == "Gradient Boosting":
            feat_importances_gradient = pd.DataFrame(regressor_gradient.feature_importances_, index=feats.columns, columns=["importance"])
            feat_importances_gradient.sort_values(by="importance", ascending=False, inplace=True)
            ax = feat_importances_gradient.plot(kind="bar", figsize=(8, 6))
            plt.title("Feature Importances - Gradient Boosting")
            plt.xlabel("Features")
            plt.ylabel("Importance")
            return ax.get_figure()

    st.write("Score: ", train_model(modele_choisi))
    st.pyplot(graph_model(modele_choisi))
    
    st.write("### Comparaison des métriques")
    st.write("Nous affichons dans un DataFrame un tableau comparatif de nos métriques.")
    st.dataframe(metrique.head())
    st.write("Le modèle le plus performant que nous retenons est le Random Forest, sur lequel nous décidons d'appliquer les hyperparamètres afin de réhausser le score.")

    st.write("### Hyperparamètres")
    st.write("Nous obtenons un score meilleur sur train grâce aux hyperparamètres. Ce qui nous permet de supprimer l'overfitting.")
    st.write("Score Train = 0.595")
    st.write("Score Test = 0.523")
    code_hyper = '''
    param_grid = {
    'n_estimators': [80],  # Nombre d'arbres dans la forêt
    'max_depth': [20],  # Profondeur maximale de l'arbre
    'min_samples_split': [20],  # Nombre minimum d'échantillons pour diviser un nœud
    'min_samples_leaf': [5],  # Nombre minimum d'échantillons par feuille
    'max_features': ['sqrt'],  # Nombre maximum de caractéristiques à considérer pour une division}'''
    st.code(code_hyper, language='python')


#PAGE CONCLUSION
elif page == pages[3]:
    st.write("### Conclusion")
    st.write("Les scores obtenus sont relativement faibles. Nous obtenons seulement 52% de prédictions justes. De plus, nous avons eu de l'overfitting sur l'ensemble des modèles testés. Seule l'application d'hyperparamètres sur notre meilleur modèle à permis de supprimer l'overfitting.")
    st.image("image3.jpg")
    st.write("### Pistes d'améliorations")
    st.write("Pour réhausser le score et optimiser le modèle, nous pouvons jouer sur deux axes : ")
    st.markdown("- Booster les hyperparamètres")
    st.write("On pourrait accroitre le nombre d’arbres dans la forêt avec le « n_estimators », mais aussi aller plus en profondeur dans l’arbre « max_depth ». On préconise également d’autres paramètres comme « max_leaf_nodes », « min_impurity_decrease » ou encore le « ccp_alpha ». Mais attention cela necessite une machine très puissante pour faire tourner ce code.")
    st.markdown("- Ajouter de nouvelles variables")
    st.write("On pourrait ajouter au modèle d'autres variables comme la densité de population, la typologie des axes routiers et la densité de circulation. Nous pensons que ces variables dont nous ne disposons pas pourrait avoir une influence significative sur le score.")
