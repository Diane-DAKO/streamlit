import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

st.title("Base de données Iris")

st.header("1. Télécharger la base de données")
st.markdown("Vous pouvez télécharger une version locale de la base Iris.")
iris = load_iris(as_frame=True)
iris_data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_data['species'] = iris.target
iris_data['species'] = iris_data['species'].map(lambda x: iris.target_names[x])
csv = iris_data.to_csv(index=False).encode('utf-8')
st.download_button("Téléchargez la base Iris (CSV)", data=csv, file_name="iris.csv")

st.header("2. Charger une base de données locale")
uploaded_file = st.file_uploader("Téléchargez un fichier CSV ayant la colonne species", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success("Base de données chargée avec succès!")
else:
    data = iris_data 
    st.info("Utilisation de la base de données par défaut (Iris).")

st.write("Aperçu des données :", data.head())

st.header("3. Manipuler les données")
if st.checkbox("Afficher des statistiques descriptives"):
    st.write(data.describe())

if st.checkbox("Filtrer par espèce"):
    species = st.selectbox("Choisissez une espèce :", data['species'].unique())
    filtered_data = data[data['species'] == species]
    st.write(f"Données pour l'espèce {species} :", filtered_data)

st.header("4. Visualisation des données")

if st.checkbox("Afficher un pairplot (Seaborn)"):
    st.subheader("Pairplot des variables")
    fig = sns.pairplot(data, hue="species", markers=["o", "s", "D"])
    st.pyplot(fig)

if st.checkbox("Afficher un boxplot"):
    st.subheader("Distribution des caractéristiques")
    feature = st.selectbox("Choisissez une caractéristique :", iris.feature_names)
    fig, ax = plt.subplots()
    sns.boxplot(data=data, x="species", y=feature, ax=ax)
    ax.set_title(f"Boxplot de {feature}")
    st.pyplot(fig)

if st.checkbox("Afficher un histogramme"):
    st.subheader("Histogramme")
    feature = st.selectbox("Sélectionnez une caractéristique :", iris.feature_names, key="hist_feature")
    fig, ax = plt.subplots()
    sns.histplot(data[feature], kde=True, ax=ax)
    ax.set_title(f"Histogramme de {feature}")
    st.pyplot(fig)

