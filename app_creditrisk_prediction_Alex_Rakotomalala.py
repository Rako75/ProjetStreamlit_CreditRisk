import streamlit as st
import pandas as pd
import numpy as np
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

home_ownership_mapping = {'RENT': 3, 'OWN': 2, 'MORTGAGE': 0, 'OTHER': 1}
loan_intent_mapping = {'PERSONAL': 4, 'EDUCATION': 1, 'MEDICAL': 3, 'VENTURE': 5, 'HOMEIMPROVEMENT': 2, 'DEBTCONSOLIDATION': 0}
loan_grade_mapping = {'D': 3, 'B': 1, 'C': 2, 'A': 0, 'E': 4, 'F': 5, 'G': 6}
cb_person_default_on_file_mapping = {'Y': 1, 'N': 0}

df['person_home_ownership'] = df['person_home_ownership'].map(home_ownership_mapping)
df['loan_intent'] = df['loan_intent'].map(loan_intent_mapping)
df['loan_grade'] = df['loan_grade'].map(loan_grade_mapping)
df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map(cb_person_default_on_file_mapping)

X = df.drop(columns=['loan_status'])
y = df['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation des données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entraînement du modèle
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, 'arbre_decision_model.joblib')

# Streamlit UI
st.title("Prédiction du Risque de Crédit par Alex Rakotomalala")
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

# Encodage des entrées utilisateur
input_data = pd.DataFrame({
    'person_age': [age],
    'person_income': [income],
    'person_home_ownership': [home_ownership_mapping.get(home_ownership, -1)],
    'person_emp_length': [emp_length],
    'loan_intent': [loan_intent_mapping.get(loan_intent, -1)],
    'loan_grade': [loan_grade_mapping.get(loan_grade, -1)],
    'loan_amnt': [loan_amnt],
    'loan_int_rate': [loan_int_rate],
    'loan_percent_income': [loan_percent_income],
    'cb_person_default_on_file': [cb_person_default_on_file_mapping.get(default_history, -1)],
    'cb_person_cred_hist_length': [cred_hist_length]
})

# Normalisation de l'entrée
input_data = scaler.transform(input_data)

# Chargement du modèle et prédiction
model = joblib.load('arbre_decision_model.joblib')
prediction = model.predict(input_data)

# Affichage du résultat de la prédiction
st.write("### Résultat de la prédiction:")
st.write("Client à risque" if prediction[0] == 1 else "Client non risqué")

# Affichage du sous-titre
st.write("""
But : cette application permet de prédire le risque de crédit d'un client en fonction de différents critères. En saisissant des informations telles que l'âge, le revenu annuel, la durée de l'emploi, le montant du prêt, etc., l'application vous indiquera si le client présente un risque de défaut de paiement sur son crédit. L'objectif est de faciliter la prise de décision dans l'octroi de crédits.
""")

# Créer un tableau des critères de risque
criteria_df = pd.DataFrame({
    'Critère': ['Âge', 'Revenu annuel', 'Type de logement', 'Durée d\'emploi', 'Motif du prêt', 
                'Note du prêt', 'Montant du prêt', 'Taux d\'intérêt', 'Ratio prêt/revenu', 
                'Historique de défaut', 'Longueur de l\'historique de crédit'],
    'Valeur': [age, income, home_ownership, emp_length, loan_intent, loan_grade, loan_amnt, loan_int_rate, 
               loan_percent_income, default_history, cred_hist_length],
    'Risque': ['Élevé' if (age < 25 or income < 20000 or loan_int_rate > 15) else 'Faible',
               'Élevé' if income < 20000 else 'Faible',
               'Élevé' if home_ownership == 'RENT' else 'Faible',
               'Élevé' if emp_length < 2 else 'Faible',
               'Élevé' if loan_intent == 'PERSONAL' else 'Faible',
               'Élevé' if loan_grade in ['D', 'E', 'F', 'G'] else 'Faible',
               'Élevé' if loan_amnt > 20000 else 'Faible',
               'Élevé' if loan_int_rate > 15 else 'Faible',
               'Élevé' if loan_percent_income > 0.5 else 'Faible',
               'Élevé' if default_history == 'Y' else 'Faible',
               'Élevé' if cred_hist_length < 5 else 'Faible']
})

# Affichage du tableau des critères de risque
st.write("### Tableau des critères de risque")
st.dataframe(criteria_df)

# Afficher une phrase expliquant le risque
st.write("""
Le client est considéré comme étant à risque en fonction des critères suivants :
- Si un ou plusieurs critères présentent des valeurs élevées (par exemple, un revenu faible, un taux d'intérêt élevé, ou un historique de défaut), cela augmente le risque de crédit.
- La combinaison de ces facteurs permet de déterminer si le client a une probabilité plus élevée de ne pas rembourser son prêt.
""")
