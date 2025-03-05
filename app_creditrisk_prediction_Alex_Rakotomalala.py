import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Charger le modèle
model = joblib.load("arbre_decision_model.joblib")
scaler = StandardScaler()

# Charger les données
@st.cache_data
def load_data():
    df = pd.read_csv("credit_risk_dataset.csv", sep=";")
    return df

df = load_data()

# Prétraitement des données
encoder = LabelEncoder()
categorical_cols = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

df.dropna(inplace=True)
scaler.fit(df.drop(columns=['loan_status']))

# Interface utilisateur Streamlit
st.title("Détection du Risque de Crédit")
st.write("Entrez les informations du client pour obtenir une prédiction de risque de crédit.")

# Entrées utilisateur
person_age = st.number_input("Âge", min_value=18, max_value=100, value=30)
person_income = st.number_input("Revenu Annuel", min_value=1000, value=50000)
person_home_ownership = st.selectbox("Type de Logement", ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
person_emp_length = st.number_input("Années d'emploi", min_value=0, max_value=50, value=5)
loan_intent = st.selectbox("But du prêt", ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
loan_grade = st.selectbox("Note de Crédit", ['D', 'B', 'C', 'A', 'E', 'F', 'G'])
loan_amnt = st.number_input("Montant du Prêt", min_value=500, value=10000)
loan_int_rate = st.number_input("Taux d'intérêt (%)", min_value=0.0, max_value=50.0, value=10.0)
loan_percent_income = loan_amnt / person_income
cb_person_default_on_file = st.selectbox("Défaut de paiement précédent", ['Y', 'N'])
cb_person_cred_hist_length = st.number_input("Durée historique de crédit", min_value=1, value=10)

# Encodage des valeurs utilisateur
input_data = pd.DataFrame({
    'person_age': [person_age],
    'person_income': [person_income],
    'person_home_ownership': [encoder.transform([person_home_ownership])[0]],
    'person_emp_length': [person_emp_length],
    'loan_intent': [encoder.transform([loan_intent])[0]],
    'loan_grade': [encoder.transform([loan_grade])[0]],
    'loan_amnt': [loan_amnt],
    'loan_int_rate': [loan_int_rate],
    'loan_percent_income': [loan_percent_income],
    'cb_person_default_on_file': [encoder.transform([cb_person_default_on_file])[0]],
    'cb_person_cred_hist_length': [cb_person_cred_hist_length]
})

# Standardisation
input_scaled = scaler.transform(input_data)

# Prédiction
if st.button("Prédire le Risque"):
    prediction = model.predict(input_scaled)[0]
    risk = "Élevé" if prediction == 1 else "Faible"
    st.subheader(f"Risque de crédit : {risk}")

# Visualisations
st.subheader("Analyse des Données")
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df['person_age'], bins=30, kde=True, color='blue', ax=ax)
ax.set_title("Distribution de l'âge des clients")
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x=df['loan_amnt'], color='orange', ax=ax)
ax.set_title("Distribution des Montants de Prêt")
st.pyplot(fig)

st.write("Cette application permet de mieux comprendre les profils clients et leur risque de crédit.")
