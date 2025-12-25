import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import seaborn as sns
import plotly.express as px

# =============================================================================
# FONCTIONS GRAPHIQUES (Reprises de tp0fin.py)
# =============================================================================

def bar_distance(t_dist: pd.DataFrame, title):
    width = max(10, len(t_dist.columns) * 1.5)
    height = 7
    fig, ax = plt.subplots(figsize=(width, height), layout='constrained')
    
    labels = t_dist.index
    x_pos = np.arange(len(labels)) 
    
    num_items = len(labels)
    total_width = 0.8
    bar_width = total_width / num_items
    
    multiplier = 0

    for col_name, distances in t_dist.items():
        offset = bar_width * multiplier
        rects = ax.bar(x_pos + offset, distances, bar_width, label=col_name)
        ax.bar_label(
            rects, 
            padding=3, 
            fmt='%.2f',
            size=14
        )
        multiplier += 1

    ax.set_title(f"Graphe de {title}")
    ax.set_ylabel("Distance")
    
    ax.set_xticks(x_pos + total_width / 2 - bar_width / 2, labels)
    
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    return fig

def graph_dist_tab(t_dist: pd.DataFrame, title):
    """
    Crée une heatmap de la matrice de distance en utilisant Seaborn.
    """
    width = max(8, len(t_dist.columns) * 0.8)
    height = max(6, len(t_dist.index) * 0.8)
    
    fig, ax = plt.subplots(figsize=(width, height))

    sns.heatmap(
        t_dist,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        linewidths=.5,
        ax=ax
    )
    
    ax.set_title(f"Heatmap de {title}")
    plt.tight_layout()
    
    return fig

def plot_distance_scales(dist_df: pd.DataFrame, title: str):
    """
    Plots distance scales for each profile, similar to the provided image.
    Each row in dist_df is a reference profile (y-axis).
    Points are plotted at their distance (x-axis).
    """
    n_items = len(dist_df)
    # Increase height if many items
    height = max(6, n_items * 0.8)
    fig, ax = plt.subplots(figsize=(10, height))
    
    # We want the first item at the top, so we reverse the index for plotting
    items = dist_df.index[::-1]
    
    # Set Y limits and ticks
    ax.set_ylim(-0.5, n_items - 0.5)
    ax.set_yticks(range(n_items))
    ax.set_yticklabels(items)
    
    # For each reference profile (row)
    for i, ref_item in enumerate(items):
        # Get distances from this ref_item to all others
        # Note: dist_df is symmetric, so row or col doesn't matter, 
        # but logically we take the row corresponding to ref_item
        distances = dist_df.loc[ref_item]
        
        # Draw a horizontal line for this item
        ax.hlines(y=i, xmin=0, xmax=distances.max() * 1.1, alpha=0.5, linewidth=1)
        
        # Plot points
        # We can use scatter
        # x = distance, y = index i
        ax.scatter(distances, [i] * len(distances), s=50, alpha=0.8)
        
        # Add labels for the points
        for target_item, dist in distances.items():
            # Offset label slightly to avoid overlap with the dot
            ax.text(dist, i + 0.1, target_item, ha='center', va='bottom', fontsize=8, rotation=0)

    ax.set_xlabel("Distance")
    ax.set_title(f"ÉCHELLES DE DISTANCES - {title}")
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig

# =============================================================================
# MAIN APPLICATION
# =============================================================================

st.title("Analyse des Lauréats")

# 1. File Upload
file = st.file_uploader("Choisissez un fichier Excel (.xlsx)", type=["xlsx"])

if file is not None:
    try:
        # Read Excel file
        t_raw = pd.read_excel(file)
        st.success("Fichier chargé avec succès !")
        
        st.subheader("Aperçu des données brutes")
        st.dataframe(t_raw.head())

        # --- NEW: Dataset Sampling ---
        st.sidebar.header("Options de l'échantillon")
        if st.sidebar.checkbox("Travailler sur un échantillon ?", value=False):
            max_val = len(t_raw)
            sample_size = st.sidebar.number_input(
                "Taille de l'échantillon", 
                min_value=2, 
                max_value=max_val, 
                value=min(1000, max_val)
            )
            if sample_size < max_val:
                t_raw = t_raw.sample(n=sample_size, random_state=42)
                st.sidebar.success(f"Échantillon de {sample_size} lignes sélectionné.")
        # -----------------------------

        # 2. Data Cleaning (Remove NaN)
        if st.checkbox("Supprimer les lignes contenant des valeurs manquantes (NaN)"):
            t_raw = t_raw.dropna()
            st.info(f"Lignes restantes après suppression des NaN : {len(t_raw)}")
        
        # FIX: Ensure all object columns are strings to avoid type errors
        for col in t_raw.select_dtypes(include=['object', 'category']).columns:
            t_raw[col] = t_raw[col].astype(str)

        # 3. Column Selection (Like aaa.py)
        st.subheader("Sélection des variables")
        cat_cols = t_raw.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if len(cat_cols) < 2:
            st.warning("Pas assez de variables catégorielles pour l'analyse.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                row_var = st.selectbox("Variable Ligne :", cat_cols, index=0)
            with col2:
                # Try to select a different second variable by default
                default_idx = 1 if len(cat_cols) > 1 else 0
                col_var = st.selectbox("Variable Colonne :", cat_cols, index=default_idx)

            if row_var == col_var:
                st.warning("Veuillez sélectionner deux variables différentes.")
            else:
                selected_cols = [row_var, col_var]
                
                # 4. Contingency Table & Filtering (Like aaa.py)
                st.sidebar.subheader("Filtres d'analyse")
                min_row_sum = st.sidebar.slider("Min observations par ligne", 0, 100, 3)
                min_col_sum = st.sidebar.slider("Min observations par colonne", 0, 100, 3)

                # Create Contingency Table
                contingency_table = pd.crosstab(t_raw[row_var], t_raw[col_var])
                
                # Filter
                filtered_table = contingency_table.loc[contingency_table.sum(axis=1) >= min_row_sum]
                filtered_table = filtered_table.loc[:, filtered_table.sum(axis=0) >= min_col_sum]

                st.subheader("Table de Contingence (Filtrée)")
                st.dataframe(filtered_table)

                if filtered_table.empty:
                    st.error("La table est vide après filtrage. Réduisez les filtres.")
                else:
                    # Filter the raw data to match the filtered contingency table
                    # We keep rows where the values are in the filtered index/columns
                    valid_rows = filtered_table.index
                    valid_cols = filtered_table.columns
                    
                    t = t_raw[t_raw[row_var].isin(valid_rows) & t_raw[col_var].isin(valid_cols)][selected_cols]
                    
                    st.info(f"Nombre d'individus après filtrage : {len(t)}")

                    # =============================================================================
                    # LOGIQUE MCA (Adaptée de tp0fin.py)
                    # =============================================================================
                    
                    t_type = []
                    rep = []
                    rep_val = []
                    rep_codage = []
                    rep_codage_comp = []

                    i = 0
                    st.subheader("Configuration des types de variables")

                    # To store the ordered levels for ordinal variables
                    ordered_levels = {}

                    for k in t.columns:
                        unique_vals = sorted(t[k].unique())  # default alphabetical order

                        var_type = st.selectbox(
                            f"Type de la variable **{k}**",
                            options=[("Nominale", 0), ("Ordinale", 1)],
                            format_func=lambda x: x[0],
                            key=f"type_{k}"
                        )[1]  # we take the int

                        t_type.append(var_type)
                        arr = np.array(t[k])
                        rep.append(arr)
                        rep_val.append(np.unique(arr))

                        if var_type == 0:  # Nominal
                            rep_codage.append(np.eye(len(unique_vals)))
                            st.info(f"→ Variable nominale : {len(unique_vals)} modalités")
                            
                        else:  # Ordinal → let the user reorder
                            st.warning(f"⚙️ Variable ordinale détectée → vous pouvez réorganiser l'ordre des modalités")
                            
                            # Default order = current order in data
                            current_order = list(t[k].unique())
                            
                            # Let the user reorder with multiselect + arrows
                            ordered = st.multiselect(
                                f"Ordre croissant des modalités pour **{k}** (glissez-déposez)",
                                options=current_order,
                                default=current_order,
                                key=f"order_{k}"
                            )
                            
                            # If the user has not reordered everything yet
                            if len(ordered) != len(current_order):
                                st.error("Vous devez placer toutes les modalités dans l'ordre désiré.")
                                st.stop()

                            ordered_levels[k] = ordered

                            # Create rank coding according to the user-defined order
                            n_mod = len(ordered)
                            rank_coding = np.zeros((n_mod, n_mod))
                            for rank in range(n_mod):
                                rank_coding[rank, :rank+1] = 1  # cumulative coding

                            rep_codage.append(rank_coding)
                            st.success(f"→ Ordre enregistré : {' → '.join(ordered)}")

                        rep_codage_comp.append(np.eye(len(unique_vals)))
                        i += 1

                    if st.button("Lancer l'analyse"):
                        taille_c = 0
                        P_ind = [f"P{_ + 1}" for _ in range(len(rep[0]))]
                        for arr in rep_codage:
                            taille_c += len(arr[0])

                        t_codage = np.zeros(shape=(len(rep[0]), taille_c))
                        t_codage_comp = np.zeros(shape=(len(rep[0]), taille_c))

                        for rep_i in range(len(rep[0])):
                            arr = []
                            arr2 = []
                            for num_rep in range(len(rep)):
                                # Find index of value in unique values
                                idx = np.where(rep_val[num_rep] == rep[num_rep][rep_i])[0][0]
                                arr.extend(rep_codage[num_rep][idx])
                                arr2.extend(rep_codage_comp[num_rep][idx])
                            t_codage[rep_i] = np.array(arr)
                            t_codage_comp[rep_i] = np.array(arr2)

                        rep_val_flat = [item for sub in rep_val for item in sub]
                        top_headers = []
                        for idx, col_name in enumerate(t.columns):
                            top_headers.extend([col_name] * len(rep_val[idx]))
                            
                        multi_index = pd.MultiIndex.from_arrays([top_headers, rep_val_flat], names=['Variable', 'Modalité'])

                        t_codage_df = pd.DataFrame(t_codage, columns=multi_index, index=P_ind)

                        st.header("Tableau de codage")
                        st.dataframe(t_codage_df)

                        t_burt = np.matmul(np.transpose(t_codage_comp), t_codage_comp)
                        t_burt_df = pd.DataFrame(t_burt, columns=multi_index, index=multi_index)

                        st.header("Tableau de Burt")
                        st.dataframe(t_burt_df)
                        
                        # --- Calcul AFC (sur Rép 1 vs Rép 2) ---
                        st.markdown("---")
                        st.subheader("Analyse détaillée sur les 2 premières variables sélectionnées")
                        
                        if len(selected_cols) >= 2:
                            # Indices pour var 1
                            len_v1 = len(rep_val[0])
                            # Indices pour var 2
                            len_v2 = len(rep_val[1])
                            
                            # Sous-matrice de Burt (croisement V1 x V2)
                            sub_burt = t_burt[0:len_v1, len_v1:len_v1+len_v2]
                            
                            # Total observations
                            total_obs = len(t_codage)
                            
                            t_frequence = np.zeros(shape=(len_v1 + 1, len_v2 + 1))

                            for r in range(len_v1):
                                for c in range(len_v2):
                                    val = sub_burt[r, c]
                                    t_frequence[r][c] = val / total_obs
                                    t_frequence[r][-1] += val / total_obs # Marge ligne
                                    t_frequence[-1][c] += val / total_obs # Marge colonne
                            
                            t_frequence[-1][-1] = t_frequence[-1, :-1].sum()

                            rows_freq = [f"{rep_val[0][i]}" for i in range(len_v1)] + ["Total"]
                            cols_freq = [f"{rep_val[1][i]}" for i in range(len_v2)] + ["Total"]
                            
                            t_frequence_df = pd.DataFrame(t_frequence, index=rows_freq, columns=cols_freq)
                            st.header(f"Tableau de fréquence ({selected_cols[0]} vs {selected_cols[1]})")
                            st.dataframe(t_frequence_df)

                            # Profils Lignes
                            t_profils_l = np.zeros(shape=(len_v1, len_v2 + 1)) # +1 pour la marge
                            # Profils Colonnes
                            t_profils_c = np.zeros(shape=(len_v1 + 1, len_v2)) # +1 pour la marge

                            for r in range(len_v1):
                                for c in range(len_v2):
                                    if t_frequence[r, -1] > 0:
                                        t_profils_l[r, c] = t_frequence[r, c] / t_frequence[r, -1]
                                    if t_frequence[-1, c] > 0:
                                        t_profils_c[r, c] = t_frequence[r, c] / t_frequence[-1, c]

                            # Ajouter les marges
                            t_profils_l[:, -1] = t_frequence[:-1, -1] # Marge ligne (poids)
                            t_profils_c[-1, :] = t_frequence[-1, :-1] # Marge colonne (poids)

                            st.header("Tableau de Profils Lignes")
                            cols_pl = cols_freq[:-1] + ["Marge (Poids)"]
                            rows_pl = rows_freq[:-1]
                            t_profils_l_df = pd.DataFrame(t_profils_l, columns=cols_pl, index=rows_pl)
                            st.dataframe(t_profils_l_df)

                            st.header("Nuage N(I) (Lignes)")
                            # Poids est la dernière colonne
                            nuage_ni = [[row[-1], row[:-1]] for row in t_profils_l]
                            # Affichage simplifié
                            st.dataframe(pd.DataFrame({"Poids": t_profils_l[:, -1]}, index=rows_pl))

                            st.header("Tableau de Profils Colonnes")
                            cols_pc = cols_freq[:-1]
                            rows_pc = rows_freq[:-1] + ["Marge (Poids)"]
                            t_profils_c_df = pd.DataFrame(t_profils_c, columns=cols_pc, index=rows_pc)
                            st.dataframe(t_profils_c_df)
                            
                            st.header("Nuage N(J) (Colonnes)")
                            # Poids est la dernière ligne
                            nuage_nj = [[col[-1], col[:-1]] for col in t_profils_c.T]
                            st.dataframe(pd.DataFrame({"Poids": t_profils_c[-1, :]}, index=cols_pc))
                            
                            #---------------- Calcul des distances Khi-2 --------
                            # Centres (poids marginaux)
                            centre_nj = t_profils_l[:, -1] # fi.
                            centre_ni = t_profils_c[-1, :] # f.j

                            # Distances N(I)
                            dist_ni = np.zeros((len_v1, len_v1))
                            for r1 in range(len_v1):
                                for r2 in range(len_v1):
                                    # Somme pondérée par l'inverse du centre de colonne (1/f.j)
                                    # Formule: sum( (fij/fi. - fi'j/fi'.)^2 / f.j )
                                    d = 0
                                    for j in range(len_v2):
                                        if centre_ni[j] > 0:
                                            p1 = t_profils_l[r1, j]
                                            p2 = t_profils_l[r2, j]
                                            d += ((p1 - p2)**2) / centre_ni[j]
                                    dist_ni[r1, r2] = d
                            
                            # Distances N(J)
                            dist_nj = np.zeros((len_v2, len_v2))
                            for c1 in range(len_v2):
                                for c2 in range(len_v2):
                                    d = 0
                                    for i in range(len_v1):
                                        if centre_nj[i] > 0:
                                            p1 = t_profils_c[i, c1]
                                            p2 = t_profils_c[i, c2]
                                            d += ((p1 - p2)**2) / centre_nj[i]
                                    dist_nj[c1, c2] = d

                            # --- DataFrames des distances (AVEC LABELS) ---
                            dist_ni_df = pd.DataFrame(dist_ni, index=rows_pl, columns=rows_pl)
                            dist_nj_df = pd.DataFrame(dist_nj, index=cols_pc, columns=cols_pc)
                            
                            # --- Affichage des distances N(I) et N(J) ---
                            st.header("Tableau de distance N(I)")
                            st.dataframe(dist_ni_df)
                            st.pyplot(bar_distance(dist_ni_df, "Distance N(I)"))
                            #st.pyplot(graph_dist_tab(dist_ni_df, "Distance N(I)"))
                            
                            # NEW: Distance Scale Graph for N(I)
                            st.subheader("Échelles de Distances - Profils Lignes N(I)")
                            st.pyplot(plot_distance_scales(dist_ni_df, "Profils Lignes N(I)"))

                            st.header("Tableau de distance N(J)")
                            st.dataframe(dist_nj_df)
                            st.pyplot(bar_distance(dist_nj_df, "Distance N(J)"))
                            st.pyplot(graph_dist_tab(dist_nj_df, "Distance N(J)"))
                            
                            # NEW: Distance Scale Graph for N(J)
                            st.subheader("Échelles de Distances - Profils Colonnes N(J)")
                            st.pyplot(plot_distance_scales(dist_nj_df, "Profils Colonnes N(J)"))

                        #---------------- Calcul de dissimilarité (sur codage complet) --------
                        st.markdown("---")
                        st.subheader("Distances globales (Dissimilarité/Similarité sur individus)")
                        
                        # Attention: sur un gros fichier, ça peut être très lent (N x N).
                        # On va mettre une limite ou un warning.
                        if len(t_codage) > 500:
                            st.warning(f"Attention: Vous avez {len(t_codage)} individus. Le calcul de la matrice de distance {len(t_codage)}x{len(t_codage)} peut être long et faire planter l'application.")
                            if not st.checkbox("Calculer quand même les distances entre individus ?"):
                                st.stop()

                        taille_c = len(t_codage)
                        t_distance = np.zeros(shape=(taille_c, taille_c))

                        # Optimisation vectorisée possible, mais gardons la logique de boucle pour l'instant ou légère optimisation
                        # La logique originale: sum((abs(p1 - p2)))/ len(p1)
                        # C'est la distance de Manhattan normalisée par le nombre de variables (si len(p1) est constant)
                        
                        # Version un peu plus rapide que la double boucle python pure
                        # t_codage est (N, M)
                        # On peut utiliser cdist de scipy si dispo, sinon boucle optimisée
                        from scipy.spatial.distance import cdist
                        # cityblock est Manhattan. On divise ensuite par le nombre de colonnes ? 
                        # Dans le code original: len(p1) est le nombre de colonnes du tableau disjonctif complet.
                        
                        dist_matrix = cdist(t_codage, t_codage, metric='cityblock')
                        t_distance = dist_matrix / t_codage.shape[1]

                        t_distance_df = pd.DataFrame(t_distance, index=P_ind, columns=P_ind)
                        
                        st.header("Tableau de distance (dissimilarité)")
                        st.dataframe(t_distance_df)
                        #st.pyplot(graph_dist_tab(t_distance_df, "Distance (dissimilarité)"))
                        
                        st.header("Tableau de distance (similarité)")
                        t_similarite_df = 1 - t_distance_df
                        st.dataframe(t_similarite_df)
                        #st.pyplot(graph_dist_tab(t_similarite_df, "Distance (similarité)"))

    except Exception as e:
        st.error(f"Une erreur s'est produite : {e}")

# =============================================================================
# CARTE DE L'EUROPE (Nouvelle fonctionnalité)
# =============================================================================
if file is not None and 't_raw' in locals():
    st.markdown("---")
    st.header("Carte des Lauréats nés en Europe")
    
    try:
        # Check if 'Born country' exists
        if 'Born country' in t_raw.columns:
            # Count laureates by country
            # We use t_raw (cleaned or not based on user choice)
            df_map = t_raw['Born country'].value_counts().reset_index()
            df_map.columns = ['Country', 'Count']
            
            # Create Plotly Choropleth Map
            fig_map = px.choropleth(
                df_map,
                locations='Country',
                locationmode='country names',
                color='Count',
                hover_name='Country',
                scope='europe', # Focus on Europe
                color_continuous_scale='Viridis',
                title='Nombre de lauréats par pays de naissance (Europe)'
            )
            
            st.plotly_chart(fig_map)
        else:
            st.warning("La colonne 'Born country' est introuvable pour générer la carte.")
            
    except Exception as e:
        st.error(f"Erreur lors de la génération de la carte : {e}")

