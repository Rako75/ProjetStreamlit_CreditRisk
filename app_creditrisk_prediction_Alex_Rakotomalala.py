import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Charger le modèle
model = joblib.load('arbre_decision_model.joblib')

# Charger le dataset
@st.cache_data
def load_data():
    df = pd.read_csv("credit_risk_dataset.csv", sep=";")
    df.fillna(df.median(numeric_only=True), inplace=True)
    encoder = LabelEncoder()
    for col in ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']:
        df[col] = encoder.fit_transform(df[col])
    return df

df = load_data()

# Fonction de prédiction
def predict_risk(features):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform([features])
    prediction = model.predict(scaled_features)
    return "Risque élevé" if prediction[0] == 1 else "Risque faible"

# Interface Streamlit
st.title("Prédiction du Risque de Crédit")
st.write("Entrez les caractéristiques du client pour obtenir une prédiction.")

# Entrée utilisateur
person_age = st.number_input("Âge", min_value=18, max_value=100, value=30)
person_income = st.number_input("Revenu annuel", min_value=1000, max_value=1000000, value=50000)
person_home_ownership = st.selectbox("Type de logement", ["RENT", "OWN", "MORTGAGE", "OTHER"])
person_emp_length = st.number_input("Années d'emploi", min_value=0, max_value=50, value=5)
loan_intent = st.selectbox("But du prêt", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
loan_grade = st.selectbox("Grade du prêt", ["D", "B", "C", "A", "E", "F", "G"])
loan_amnt = st.number_input("Montant du prêt", min_value=100, max_value=50000, value=5000)
loan_int_rate = st.number_input("Taux d'intérêt", min_value=0.0, max_value=50.0, value=10.0)
loan_percent_income = st.number_input("Pourcentage du revenu consacré au prêt", min_value=0.0, max_value=1.0, value=0.2)
cb_person_default_on_file = st.selectbox("Défaut de paiement antérieur", ["Y", "N"])
cb_person_cred_hist_length = st.number_input("Longueur de l'historique de crédit", min_value=0, max_value=30, value=5)

# Encodage des valeurs catégoriques
encoder = LabelEncoder()
person_home_ownership = encoder.fit_transform([person_home_ownership])[0]
loan_intent = encoder.fit_transform([loan_intent])[0]
loan_grade = encoder.fit_transform([loan_grade])[0]
cb_person_default_on_file = encoder.fit_transform([cb_person_default_on_file])[0]

# Prédiction
if st.button("Prédire le Risque"):
    features = [
        person_age, person_income, person_home_ownership, person_emp_length,
        loan_intent, loan_grade, loan_amnt, loan_int_rate,
        loan_percent_income, cb_person_default_on_file, cb_person_cred_hist_length
    ]
    prediction = predict_risk(features)
    st.write(f"Résultat de la prédiction : {prediction}")

# Visualisations interactives
st.subheader("Analyse Exploratoire des Données")

# Histogramme
st.write("### Distribution des variables clés")
fig, ax = plt.subplots()
sns.histplot(df['person_age'], bins=30, kde=True, color='blue', ax=ax)
st.pyplot(fig)

# Heatmap
st.write("### Corrélation entre les variables")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
st.pyplot(fig)
