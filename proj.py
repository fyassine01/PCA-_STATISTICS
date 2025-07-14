import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import streamlit as st
import io
import base64

# Configuration de la page
st.set_page_config(
    page_title="Outil d'Analyse ACP",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ajout de CSS personnalisé pour le style
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1 {
        color: #1E88E5;
    }
    h2 {
        color: #388E3C;
        margin-top: 2rem;
    }
    .interpretation {
        background-color: #f8f9fa;
        border-left: 4px solid #1E88E5;
        padding: 1rem;
        margin: 1rem 0;
        color: #333;
    }
    .interpretation h3 {
        color: #2C3E50;
        margin-top: 1rem;
        margin-bottom: 0.75rem;
    }
    .interpretation ul, .interpretation ol {
        margin-left: 1.5rem;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# Titre et introduction
st.title("Analyse en Composantes Principales (ACP) avec Interprétation Automatisée")
st.markdown("""
Cette application effectue une Analyse en Composantes Principales (ACP) sur vos données et fournit :
1. Un graphique d'éboulis (Scree Plot) montrant la variance expliquée par chaque composante principale
2. Un Biplot montrant la projection des individus et des variables sur CP1 et CP2
3. Une interprétation automatisée des résultats de l'ACP

Téléchargez votre fichier CSV pour commencer.
""")

# Barre latérale pour le téléchargement de fichiers et les paramètres
with st.sidebar:
    st.header("Données d'entrée")
    uploaded_file = st.file_uploader("Télécharger un fichier CSV", type=["csv"])
    
    st.header("Paramètres de l'ACP")
    scale_data = st.checkbox("Standardiser les données (recommandé)", value=True)
    n_components = st.slider("Nombre maximal de composantes à calculer", min_value=2, max_value=10, value=5)
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.header("Sélection des variables")
        
        # Affichage d'un échantillon des données
        st.subheader("Échantillon de données")
        st.dataframe(df.head(5))
        
        # Sélection de la variable cible (optionnel)
        all_columns = df.columns.tolist()
        color_by = st.selectbox("Colorer les points par (optionnel)", ["Aucun"] + all_columns)
        
        # Sélection des colonnes pour l'ACP
        st.subheader("Sélectionner les colonnes pour l'analyse")
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_columns = st.multiselect("Sélectionner les colonnes pour l'analyse", 
                                        numeric_columns, 
                                        default=numeric_columns[:min(5, len(numeric_columns))])

# Fonction d'interprétation IA
def interpret_pca_results(pca, feature_names, loadings, explained_variance_ratio, df_pca):
    """Générer une interprétation des résultats de l'ACP"""
    
    # Interprétation de la variance expliquée
    var_interpretation = f"""### Analyse de la variance expliquée
    
La première composante principale (CP1) explique {explained_variance_ratio[0]*100:.2f}% de la variance totale des données.
La deuxième composante principale (CP2) explique {explained_variance_ratio[1]*100:.2f}% supplémentaires de la variance.

Ensemble, CP1 et CP2 capturent {(explained_variance_ratio[0] + explained_variance_ratio[1])*100:.2f}% de l'information totale dans le jeu de données.
"""

    # Recherche d'un coude dans le graphique d'éboulis
    diffs = np.diff(explained_variance_ratio)
    elbow_detected = False
    elbow_component = 0
    
    for i in range(len(diffs)):
        if i > 0 and diffs[i-1] > 0.1 and diffs[i] < 0.05:
            elbow_detected = True
            elbow_component = i + 1
            break
    
    if elbow_detected:
        var_interpretation += f"\nUn coude apparaît à la composante {elbow_component}, suggérant que les {elbow_component} premières composantes peuvent être suffisantes pour représenter efficacement les données."
    else:
        var_interpretation += "\nAucun coude clair n'est détecté dans le graphique d'éboulis. Considérez l'examen de la variance expliquée cumulée pour déterminer combien de composantes conserver."
    
    # Interpréter les contributions des caractéristiques
    feature_interpretation = "### Contributions des caractéristiques\n\n"
    
    # Interprétation CP1
    pc1_loading = loadings[:, 0]
    sorted_idx = np.argsort(np.abs(pc1_loading))[::-1]
    top_features_pc1 = [(feature_names[idx], pc1_loading[idx]) for idx in sorted_idx[:3]]
    
    feature_interpretation += "**Composante Principale 1 (CP1)** est le plus fortement influencée par :\n\n"
    for feature, loading in top_features_pc1:
        direction = "positivement" if loading > 0 else "négativement"
        feature_interpretation += f"* {feature} (corrélée {direction}, coefficient = {loading:.3f})\n"
    
    # Interprétation CP2
    pc2_loading = loadings[:, 1]
    sorted_idx = np.argsort(np.abs(pc2_loading))[::-1]
    top_features_pc2 = [(feature_names[idx], pc2_loading[idx]) for idx in sorted_idx[:3]]
    
    feature_interpretation += "\n**Composante Principale 2 (CP2)** est le plus fortement influencée par :\n\n"
    for feature, loading in top_features_pc2:
        direction = "positivement" if loading > 0 else "négativement"
        feature_interpretation += f"* {feature} (corrélée {direction}, coefficient = {loading:.3f})\n"
    
    # Recherche de clusters ou modèles dans le biplot
    cluster_interpretation = "### Analyse des motifs\n\n"
    
    # Vérification des valeurs aberrantes
    distances = np.sqrt(df_pca[:, 0]**2 + df_pca[:, 1]**2)
    threshold = np.mean(distances) + 2 * np.std(distances)
    outliers = np.where(distances > threshold)[0]
    
    if len(outliers) > 0:
        cluster_interpretation += f"Il y a {len(outliers)} valeurs aberrantes potentielles détectées dans le biplot, qui sont des observations éloignées du centre du graphique.\n\n"
    
    # Vérification des modèles de regroupement (approche très simple)
    from sklearn.cluster import KMeans
    
    # Essayer 2-4 clusters et vérifier le score silhouette
    from sklearn.metrics import silhouette_score
    
    best_score = -1
    best_n_clusters = 2
    
    if len(df_pca) > 10:  # Besoin d'un nombre raisonnable d'échantillons
        for n_clusters in range(2, min(5, len(df_pca) // 5 + 1)):
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                labels = kmeans.fit_predict(df_pca[:, :2])
                
                if len(np.unique(labels)) > 1:  # S'assurer d'avoir au moins 2 clusters
                    score = silhouette_score(df_pca[:, :2], labels)
                    
                    if score > best_score:
                        best_score = score
                        best_n_clusters = n_clusters
            except:
                pass
        
        if best_score > 0.3:
            cluster_interpretation += f"Les données semblent former approximativement {best_n_clusters} groupes dans l'espace CP1-CP2 (score silhouette = {best_score:.2f}), suggérant des sous-groupes distincts dans vos données.\n\n"
        elif best_score > 0:
            cluster_interpretation += f"Il pourrait y avoir un certain regroupement dans les données, mais les clusters ne sont pas fortement séparés (score silhouette = {best_score:.2f}).\n\n"
        else:
            cluster_interpretation += "Aucun modèle de regroupement clair n'est évident dans l'espace ACP.\n\n"
    
    # Résumé global
    summary = "### Résumé\n\n"
    summary += f"La dimensionnalité de vos données peut être réduite à {elbow_component if elbow_detected else 2} composantes principales tout en conservant la plupart des informations importantes. "
    
    # Identifier les caractéristiques corrélées
    correlation_summary = ""
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            if abs(np.dot(loadings[i], loadings[j])) > 0.7:
                correlation_summary += f"* {feature_names[i]} et {feature_names[j]} semblent être fortement corrélés.\n"
    
    if correlation_summary:
        summary += "\n\nCaractéristiques corrélées identifiées :\n" + correlation_summary
    
    # Recommandations finales
    recommendations = "### Recommandations\n\n"
    recommendations += "Basées sur les résultats de l'ACP :\n\n"
    
    if explained_variance_ratio[0] > 0.5:
        recommendations += "* CP1 explique une majorité de la variance, suggérant que les données pourraient être efficacement représentées dans une dimension inférieure.\n"
    
    if elbow_detected:
        recommendations += f"* Envisagez d'utiliser les {elbow_component} premières composantes principales pour une analyse ou une modélisation ultérieure.\n"
    else:
        cumulative_var = np.cumsum(explained_variance_ratio)
        for i, var in enumerate(cumulative_var):
            if var > 0.8:
                recommendations += f"* Pour conserver 80% de la variance, utilisez les {i+1} premières composantes principales.\n"
                break
    
    if len(outliers) > 0:
        recommendations += "* Examinez les valeurs aberrantes identifiées dans le biplot car elles pourraient représenter des anomalies importantes ou des problèmes de qualité des données.\n"
    
    # Combiner toutes les interprétations
    full_interpretation = var_interpretation + "\n\n" + feature_interpretation + "\n\n" + cluster_interpretation + "\n\n" + summary + "\n\n" + recommendations
    
    return full_interpretation

# Générer un lien téléchargeable pour le graphique
def get_plot_download_link(fig, filename="plot.png", text="Télécharger le graphique"):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Fonction principale d'analyse ACP
def run_pca_analysis(df, selected_columns, color_by, scale_data, n_components):
    """Exécuter l'analyse ACP et générer des graphiques"""
    
    # Préparer les données pour l'ACP
    X = df[selected_columns].copy()
    
    # Gérer les valeurs manquantes (remplacer par la moyenne)
    X = X.fillna(X.mean())
    
    # Mettre à l'échelle les données si demandé
    if scale_data:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.values
    
    # Exécuter l'ACP
    pca = PCA(n_components=min(n_components, len(selected_columns)))
    pc_scores = pca.fit_transform(X_scaled)
    
    # Créer un DataFrame pour faciliter le traçage
    pc_df = pd.DataFrame(data=pc_scores, columns=[f'CP{i+1}' for i in range(pc_scores.shape[1])])
    
    # Ajouter la colonne de couleur si spécifiée
    if color_by != "Aucun":
        pc_df['color'] = df[color_by].values
    
    # Calculer les chargements (corrélations entre variables et composantes principales)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    # Créer des graphiques
    col1, col2 = st.columns(2)
    
    # 1. Graphique d'éboulis
    with col1:
        st.header("Graphique d'éboulis")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        
        # Diagramme à barres de la variance expliquée
        ax1.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
                pca.explained_variance_ratio_, alpha=0.6, color='skyblue', 
                label='Variance expliquée individuelle')
        
        # Graphique en ligne de la variance expliquée cumulée
        ax1.step(range(1, len(pca.explained_variance_ratio_) + 1), 
                 np.cumsum(pca.explained_variance_ratio_), where='mid',
                 label='Variance expliquée cumulée', color='red', marker='o')
        
        # Ajouter une ligne horizontale à 80% et 95%
        ax1.axhline(y=0.8, color='g', linestyle='--', alpha=0.5, label='Seuil de variance 80%')
        ax1.axhline(y=0.95, color='purple', linestyle='--', alpha=0.5, label='Seuil de variance 95%')
        
        # Ajouter des étiquettes et une légende
        ax1.set_xlabel('Composantes Principales')
        ax1.set_ylabel('Ratio de Variance Expliquée')
        ax1.set_title('Graphique d\'éboulis : Variance expliquée par composante principale')
        ax1.set_xticks(range(1, len(pca.explained_variance_ratio_) + 1))
        ax1.set_xticklabels([f'CP{i}' for i in range(1, len(pca.explained_variance_ratio_) + 1)])
        ax1.legend(loc='best')
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # Afficher le graphique
        st.pyplot(fig1)
        
        # Tableau détaillé de la variance
        var_df = pd.DataFrame({
            'Composante Principale': [f'CP{i+1}' for i in range(len(pca.explained_variance_ratio_))],
            'Valeur propre': pca.explained_variance_,
            'Variance expliquée (%)': pca.explained_variance_ratio_ * 100,
            'Variance cumulée (%)': np.cumsum(pca.explained_variance_ratio_) * 100
        })
        
        st.subheader("Informations détaillées sur la variance")
        st.dataframe(var_df.style.format({
            'Valeur propre': '{:.3f}',
            'Variance expliquée (%)': '{:.2f}%',
            'Variance cumulée (%)': '{:.2f}%'
        }))
        
        # Lien de téléchargement
        st.markdown(get_plot_download_link(fig1, "graphique_eboulis.png", "📥 Télécharger le graphique d'éboulis"), unsafe_allow_html=True)
    
    # 2. Biplot
    with col2:
        st.header("Biplot")
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        
        # Configurer les couleurs
        if color_by != "Aucun":
            if pd.api.types.is_numeric_dtype(df[color_by]):
                scatter = ax2.scatter(pc_df['CP1'], pc_df['CP2'], c=pc_df['color'], 
                               cmap='viridis', alpha=0.7, s=50)
                plt.colorbar(scatter, ax=ax2, label=color_by)
            else:
                categories = df[color_by].unique()
                cmap = plt.get_cmap('tab10', len(categories))
                
                for i, category in enumerate(categories):
                    mask = pc_df['color'] == category
                    ax2.scatter(pc_df.loc[mask, 'CP1'], pc_df.loc[mask, 'CP2'], 
                              label=str(category), color=cmap(i), alpha=0.7, s=50)
                
                ax2.legend(title=color_by)
        else:
            ax2.scatter(pc_df['CP1'], pc_df['CP2'], alpha=0.7, s=50)
        
        # Tracer les vecteurs de caractéristiques
        for i, feature in enumerate(selected_columns):
            ax2.arrow(0, 0, loadings[i, 0] * 5, loadings[i, 1] * 5, 
                    head_width=0.2, head_length=0.2, fc='red', ec='red')
            ax2.text(loadings[i, 0] * 5.2, loadings[i, 1] * 5.2, feature, color='red')
        
        # Ajouter des étiquettes et un titre
        ax2.set_xlabel(f'Composante Principale 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax2.set_ylabel(f'Composante Principale 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax2.set_title('Biplot ACP : Projection des observations et des variables')
        
        # Ajouter des lignes de grille
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
        ax2.grid(True, linestyle='--', alpha=0.3)
        
        # Rendre le graphique joli
        ax2.set_aspect('equal')
        
        # Afficher le graphique
        st.pyplot(fig2)
        
        # Lien de téléchargement
        st.markdown(get_plot_download_link(fig2, "biplot.png", "📥 Télécharger le Biplot"), unsafe_allow_html=True)
    
    # Créer un tableau de chargements
    loadings_df = pd.DataFrame(
        loadings, 
        columns=[f'CP{i+1}' for i in range(loadings.shape[1])],
        index=selected_columns
    )
    
    # Interprétation IA
    st.header("Interprétation des résultats de l'ACP")
    
    with st.spinner("Génération de l'interprétation en cours..."):
        interpretation = interpret_pca_results(
            pca, 
            selected_columns, 
            loadings, 
            pca.explained_variance_ratio_,
            pc_scores
        )
    
    st.markdown(f'''
    <div class='interpretation'>
        {interpretation}
    </div>
    ''', unsafe_allow_html=True)
    
    # Afficher les chargements
    st.header("Contributions des variables")
    st.dataframe(loadings_df.style.background_gradient(cmap='coolwarm', axis=1))
    
    # Options de téléchargement
    st.header("Exporter les résultats")
    
    # Télécharger les scores
    scores_csv = pc_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Télécharger les scores CP (CSV)",
        data=scores_csv,
        file_name="scores_acp.csv",
        mime="text/csv"
    )
    
    # Télécharger les chargements
    loadings_csv = loadings_df.to_csv().encode('utf-8')
    st.download_button(
        label="Télécharger les contributions des variables (CSV)",
        data=loadings_csv,
        file_name="contributions_acp.csv",
        mime="text/csv"
    )

# Exécuter l'analyse si les données sont téléchargées et les colonnes sélectionnées
if uploaded_file is not None and 'selected_columns' in locals() and selected_columns:
    if len(selected_columns) < 2:
        st.error("Veuillez sélectionner au moins 2 colonnes pour l'analyse ACP.")
    else:
        run_pca_analysis(df, selected_columns, color_by, scale_data, n_components)
else:
    # Afficher une sortie d'exemple si aucune donnée n'est téléchargée
    st.info("Téléchargez un fichier CSV et sélectionnez des colonnes pour effectuer l'analyse ACP.")
    
    # Ajouter une option de jeu de données d'exemple
    if st.button("Charger un jeu de données d'exemple"):
        # Créer un jeu de données d'exemple simple
        np.random.seed(42)
        n_samples = 100
        
        # Créer quelques caractéristiques corrélées
        x1 = np.random.normal(0, 1, n_samples)
        x2 = x1 * 0.8 + np.random.normal(0, 0.2, n_samples)
        x3 = np.random.normal(0, 1, n_samples)
        x4 = x3 * 0.7 + np.random.normal(0, 0.3, n_samples)
        x5 = np.random.normal(0, 1, n_samples)
        
        # Créer une variable catégorielle pour la coloration
        group = np.concatenate([np.zeros(n_samples//2), np.ones(n_samples//2)])
        
        # Créer un dataframe
        example_df = pd.DataFrame({
            'Variable1': x1,
            'Variable2': x2,
            'Variable3': x3,
            'Variable4': x4,
            'Variable5': x5,
            'Groupe': group
        })
        
        # Exécuter l'ACP sur les données d'exemple
        run_pca_analysis(
            example_df, 
            ['Variable1', 'Variable2', 'Variable3', 'Variable4', 'Variable5'], 
            'Groupe', 
            True, 
            5
        )

# Ajouter un pied de page
st.markdown("""
---
### Comment utiliser cet outil :

1. **Téléchargez un fichier CSV** contenant vos données.
2. **Sélectionnez les variables** pour l'analyse ACP.
3. **Optionnel** : Choisissez une colonne pour coder en couleur les points dans le biplot.
4. **Examinez les résultats** :
   - Le graphique d'éboulis montre combien de variance chaque composante principale explique
   - Le Biplot montre comment les observations et les variables sont positionnées dans l'espace CP
   - L'interprétation IA explique les principales conclusions de votre analyse ACP
5. **Téléchargez** les graphiques et les données pour vos rapports et analyses supplémentaires.

L'ACP est une technique de réduction de dimensionnalité qui transforme des variables corrélées en un ensemble plus petit de variables non corrélées appelées composantes principales. Cela aide à visualiser des modèles dans des données à haute dimension.
""")