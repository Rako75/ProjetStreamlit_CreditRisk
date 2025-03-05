import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Charger le modèle et le scaler sauvegardés
model = joblib.load('arbre_decision_model.joblib')
scaler = joblib.load('scaler.pkl')

# Interface utilisateur
st.title("Prédiction du statut de prêt")

# Saisie des données utilisateur
age = st.number_input("Âge", min_value=18, max_value=100, value=30)
income = st.number_input("Revenu", min_value=0, max_value=1000000, value=50000)
loan_amount = st.number_input("Montant du prêt", min_value=0, max_value=1000000, value=10000)
int_rate = st.number_input("Taux d'intérêt", min_value=0.0, max_value=50.0, value=5.0)
emp_length = st.number_input("Ancienneté de l'emploi (en années)", min_value=0, max_value=50, value=5)
home_ownership = st.selectbox("Type de logement", ["OWN", "MORTGAGE", "RENT"])
loan_intent = st.selectbox("Intentions du prêt", ["PERSONAL", "DEBTCONSOLIDATION", "EDUCATION"])
loan_grade = st.selectbox("Grade du prêt", ["A", "B", "C", "D", "E", "F", "G"])
credit_hist_length = st.number_input("Longueur de l'historique de crédit (en années)", min_value=0, max_value=50, value=5)

# Colonnes manquantes : Ajout de valeurs par défaut pour les colonnes qui ne sont pas saisies
cb_person_default_on_file = 0  # Par exemple, un défaut (0) par défaut
loan_percent_income = income / loan_amount if loan_amount != 0 else 0  # Calcul d'un pourcentage par défaut

# Préparer les données pour la prédiction, en ajoutant les colonnes manquantes
data = pd.DataFrame({
    'person_age': [age],
    'person_income': [income],
    'loan_amnt': [loan_amount],
    'loan_int_rate': [int_rate],
    'person_emp_length': [emp_length],
    'person_home_ownership': [home_ownership],
    'loan_intent': [loan_intent],
    'loan_grade': [loan_grade],
    'cb_person_cred_hist_length': [credit_hist_length],
    'cb_person_default_on_file': [cb_person_default_on_file],  # Ajout de la colonne manquante
    'loan_percent_income': [loan_percent_income]  # Ajout de la colonne manquante
})

# Traitement des valeurs manquantes avec SimpleImputer
imputer = SimpleImputer(strategy="median")

# Liste des colonnes numériques
numeric_columns = ['person_emp_length', 'loan_int_rate']

# Appliquer l'imputation pour les colonnes numériques
data[numeric_columns] = imputer.fit_transform(data[numeric_columns])

# Encoder les variables catégorielles
encoder = LabelEncoder()
data['person_home_ownership'] = encoder.fit(["OWN", "MORTGAGE", "RENT"]).transform([data['person_home_ownership'][0]])
data['loan_intent'] = encoder.fit(["PERSONAL", "DEBTCONSOLIDATION", "EDUCATION"]).transform([data['loan_intent'][0]])
data['loan_grade'] = encoder.fit(["A", "B", "C", "D", "E", "F", "G"]).transform([data['loan_grade'][0]])
data['cb_person_default_on_file'] = encoder.fit(["0", "1"]).transform([str(data['cb_person_default_on_file'][0])])

# Appliquer la normalisation
data_normalized = scaler.transform(data)

# Bouton pour effectuer la prédiction
if st.button("Prédire le statut du prêt"):
    # Prédire le statut du prêt
    prediction = model.predict(data_normalized)
    if prediction == 1:
        st.write("Le client est à risque de défaut.")
    else:
        st.write("Le client n'est pas à risque de défaut.")
