import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Chargement des données
df = pd.read_csv("credit_risk_dataset.csv", sep=";")

# Prétraitement
df['loan_int_rate'].fillna(df['loan_int_rate'].median(), inplace=True)
df['person_emp_length'].fillna(df['person_emp_length'].median(), inplace=True)

encoder = LabelEncoder()
df['person_home_ownership'] = encoder.fit_transform(df['person_home_ownership'])
df['loan_intent'] = encoder.fit_transform(df['loan_intent'])
df['loan_grade'] = encoder.fit_transform(df['loan_grade'])
df['cb_person_default_on_file'] = encoder.fit_transform(df['cb_person_default_on_file'])

X = df.drop(columns=['loan_status'])
y = df['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entraînement du modèle
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, 'credit_risk_model.joblib')

# Streamlit UI
st.title("Prédiction du Risque de Crédit")
st.sidebar.header("Entrez les informations du client")

# Entrée utilisateur
age = st.sidebar.slider("Âge", min_value=18, max_value=100, value=30)
income = st.sidebar.number_input("Revenu annuel", min_value=0, value=50000)
home_ownership = st.sidebar.selectbox("Type de logement", ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
emp_length = st.sidebar.slider("Durée d'emploi (années)", min_value=0, max_value=50, value=5)
loan_intent = st.sidebar.selectbox("Motif du prêt", ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
loan_grade = st.sidebar.selectbox("Note du prêt", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
loan_amnt = st.sidebar.number_input("Montant du prêt", min_value=500, max_value=50000, value=10000)
loan_int_rate = st.sidebar.slider("Taux d'intérêt (%)", min_value=0.0, max_value=40.0, value=10.0)
loan_percent_income = st.sidebar.slider("Ratio prêt/revenu", min_value=0.0, max_value=1.0, value=0.2)
default_history = st.sidebar.selectbox("Historique de défaut de paiement", ['Y', 'N'])
cred_hist_length = st.sidebar.slider("Longueur de l'historique de crédit", min_value=0, max_value=30, value=10)

# Encodage des entrées
input_data = pd.DataFrame({
    'person_age': [age],
    'person_income': [income],
    'person_home_ownership': [encoder.transform([home_ownership])[0]],
    'person_emp_length': [emp_length],
    'loan_intent': [encoder.transform([loan_intent])[0]],
    'loan_grade': [encoder.transform([loan_grade])[0]],
    'loan_amnt': [loan_amnt],
    'loan_int_rate': [loan_int_rate],
    'loan_percent_income': [loan_percent_income],
    'cb_person_default_on_file': [encoder.transform([default_history])[0]],
    'cb_person_cred_hist_length': [cred_hist_length]
})

input_data = scaler.transform(input_data)

# Prédiction
model = joblib.load('credit_risk_model.joblib')
prediction = model.predict(input_data)
st.write("### Résultat de la prédiction:")
st.write("Client à risque" if prediction[0] == 1 else "Client non risqué")

# Visualisation des données
st.subheader("Exploration des Données")
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(df['person_age'], bins=30, kde=True, ax=ax)
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(x=df["loan_amnt"], ax=ax)
st.pyplot(fig)
