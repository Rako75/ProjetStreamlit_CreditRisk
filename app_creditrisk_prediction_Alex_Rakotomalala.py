import streamlit as st
import pandas as pd
import joblib

# Chargement des données
df = pd.read_csv("credit_risk_dataset.csv", sep=";")

# Détermination des seuils de risque
seuil_revenu = df['person_income'].quantile([0.25, 0.75])
seuil_pret = df['loan_amnt'].quantile([0.25, 0.75])

# Streamlit UI
st.title("Prédiction du Risque de Crédit")
st.sidebar.header("Entrez les informations du client")

# Entrée utilisateur
age = st.sidebar.slider("Âge", min_value=18, max_value=100, value=30)
income = st.sidebar.number_input("Revenu annuel", min_value=0, value=50000)
loan_amnt = st.sidebar.number_input("Montant du prêt", min_value=500, max_value=50000, value=10000)

# Chargement du modèle et prédiction
model = joblib.load('arbre_decision_model.joblib')
prediction = model.predict([[age, income, loan_amnt]])

# Affichage du résultat de la prédiction
st.write("### Résultat de la prédiction:")
st.write("Client à risque" if prediction[0] == 1 else "Client non risqué")

# Affichage du tableau des seuils
data = {
    "Critère": ["Revenu Annuel ($)", "Montant du prêt ($)"],
    "Limite Non Risqué": [seuil_revenu[0.25], seuil_pret[0.25]],
    "Limite À Risque": [seuil_revenu[0.75], seuil_pret[0.75]],
    "Valeur du Client": [income, loan_amnt],
    "Statut": ["À Risque" if income < seuil_revenu[0.25] or loan_amnt > seuil_pret[0.75] else "Non Risqué"]
}

df_result = pd.DataFrame(data)
st.table(df_result)
