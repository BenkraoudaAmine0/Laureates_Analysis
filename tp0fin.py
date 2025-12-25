import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import seaborn as sns  # Ajout de seaborn

file = st.file_uploader("Choisissez un fichier.", type="csv")
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
            fmt='%.2f',  # Format à 2 décimales
            size=14       # Réduire la taille de la police
        )
        multiplier += 1

    ax.set_title(f"Graphe de {title}")
    ax.set_ylabel("Distance")
    
    ax.set_xticks(x_pos + total_width / 2 - bar_width / 2, labels)
    
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    return fig

# =============================================================================
# FONCTION DE GRAPHIQUE (MISE À JOUR)
# =============================================================================
def graph_dist_tab(t_dist: pd.DataFrame, title):
    """
    Crée une heatmap de la matrice de distance en utilisant Seaborn.
    """
    # Définir une taille de figure dynamique (pour s'adapter aux grands tableaux)
    width = max(8, len(t_dist.columns) * 0.8)
    height = max(6, len(t_dist.index) * 0.8)
    
    # Créer la figure et l'axe
    fig, ax = plt.subplots(figsize=(width, height))

    # Dessiner la heatmap sur l'axe
    sns.heatmap(
        t_dist,
        annot=True,          # Afficher les chiffres dans les cases
        fmt=".3f",           # Formater les chiffres à 3 décimales
        cmap="Blues",
        linewidths=.5,       # Lignes fines entre les cases
        ax=ax                # Utiliser l'axe créé
    )
    
    ax.set_title(f"Heatmap de {title}")
    
    # Ajuster la mise en page pour éviter que les labels ne se chevauchent
    plt.tight_layout()
    
    # Renvoyer la figure pour Streamlit
    return fig
# =============================================================================


if file is not None:
    t = pd.read_csv(file, encoding="utf-8-sig")

    t_type = []

    rep = []
    rep_val = []
    rep_codage = []
    rep_codage_comp = []

    i = 0
    for k in t.columns:
        t_type.append(st.selectbox(f"Entrer le type de la variable {k}", (0, 1), format_func=lambda x: "nominale" if x == 0 else "ordinale"))
        arr = np.array(t[k])
        rep.append(arr)
        rep_val.append(np.unique(arr))
        temp_len =  len(np.unique(arr))

        if t_type[i] == 0:
            rep_codage.append(np.eye(temp_len))
        elif t_type[i] == 1:
            rep_codage.append(np.tril(np.ones([temp_len, temp_len])))
        
        rep_codage_comp.append(np.eye(temp_len))

        i += 1

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
            arr.extend(rep_codage[num_rep][np.where(rep_val[num_rep] == rep[num_rep][rep_i])[0][0]])
            arr2.extend(rep_codage_comp[num_rep][np.where(rep_val[num_rep] == rep[num_rep][rep_i])[0][0]])
        t_codage[rep_i] = np.array(arr)
        t_codage_comp[rep_i] = np.array(arr2)

    rep_val_flat = [item for sub in rep_val for item in sub]
    top_headers = []
    for i in [np.repeat(t.columns[_], len(rep_val[_])) for _ in range(len(rep_val))]:
        for j in i:
            top_headers.append(j)
    multi_index = pd.MultiIndex.from_arrays([top_headers, rep_val_flat], names=['', ''])

    t_codage_df = pd.DataFrame(t_codage, columns=multi_index, index=P_ind)

    st.header("Tableau de codage")
    st.dataframe(t_codage_df)

    t_burt = np.matmul(np.transpose(t_codage_comp), t_codage_comp)
    t_burt_df = pd.DataFrame(t_burt, columns=multi_index, index=multi_index)

    st.header("Tableau de Burt")
    st.dataframe(t_burt_df)
    
    # --- Calcul AFC (sur Rép 1 vs Rép 2) ---
    t_frequence = np.zeros(shape=(len(rep_val[0]) + 1, len(rep_val[1]) + 1))

    for i in range(len(t_frequence) - 1):
        freq_j_sum = 0
        for j in range(len(t_frequence[0]) - 1):
            t_frequence[i][j] = t_burt[i,j]/len(t_codage)
            t_frequence[i][-1] += t_burt[i,j]/len(t_codage)
            t_frequence[-1][j] += t_burt[i,j]/len(t_codage)
    
    # Correction: Grand total = somme des marges colonnes (ou lignes)
    t_frequence[-1][-1] = t_frequence[-1, :-1].sum()

    rows_freq = [f"f{_}j" for _ in range(len(t_frequence) - 1)]
    rows_freq.append("f.j")

    cols_freq = [f"fi{_}" for _ in range(len(t_frequence[0])-1)]
    cols_freq.append("fi.")
    t_frequence_df = pd.DataFrame(t_frequence, index=rows_freq, columns=cols_freq)
    st.header("Tableau de fréquence (Rép 1 vs Rép 2)")
    st.dataframe(t_frequence_df)

    t_profils_c = np.zeros(shape=(len(t_frequence), len(t_frequence[0]) - 1))
    t_profils_l = np.zeros(shape=(len(t_frequence) - 1, len(t_frequence[0])))

    for i in range(len(t_frequence) - 1):
        for j in range(len(t_frequence[0]) - 1):
            t_profils_l[i,j] = t_frequence[i,j] / t_frequence[i, -1]
            t_profils_c[i,j] = t_frequence[i,j] / t_frequence[-1, j]

    t_profils_l[:,-1] = t_frequence[:-1, -1] 
    t_profils_c[-1] = t_frequence[-1, :-1]

    st.header("Tableau de Profils Lignes")
    cols_pl = [f"F{_}J" for _ in range(len(t_profils_l[0]) - 1) ]
    cols_pl.append("fi.")
    rows_pl = [f"FI{_}" for _ in range(len(t_profils_l)) ] # Labels pour N(I)
    t_profils_l_df = pd.DataFrame(t_profils_l, columns=cols_pl, index=rows_pl)
    st.dataframe(t_profils_l_df)

    st.header("Nuage N(I)")
    nuage_ni = [[arr[len(arr)-1], arr[:len(arr)-1]] for arr in t_profils_l]
    st.dataframe(pd.DataFrame(nuage_ni, columns=["Poids", "Fih"], index=rows_pl))

    st.header("Tableau de Profils Colonnes")
    cols_pc = [f"F{_}J" for _ in range(len(t_profils_c[0])) ] # Labels pour N(J)
    rows_pc = [f"FI{_}" for _ in range(len(t_profils_c) - 1) ]
    rows_pc.append("f.j")
    t_profils_c_df = pd.DataFrame(t_profils_c, columns=cols_pc, index=rows_pc)
    st.dataframe(t_profils_c_df)
    
    st.header("Nuage N(J)")
    nuage_nj = [[arr[len(arr)-1], arr[:len(arr)-1]] for arr in t_profils_c.T]
    st.dataframe(pd.DataFrame(nuage_nj, columns=["Poids", "Fji"], index=cols_pc))
    
    #---------------- Calcul des distances Khi-2 --------
    centre_nj = [arr[0] for arr in nuage_ni] # Poids des lignes (fi.)
    centre_ni = [arr[0] for arr in nuage_nj] # Poids des colonnes (f.j)

    dist_ni = []
    for arr1 in nuage_ni:
        dist_ni.append([])
        for arr2 in nuage_ni:
            dist_ni[-1].append(sum((arr1[1]-arr2[1])*(arr1[1]-arr2[1])/centre_ni))
    
    dist_nj = []
    for arr1 in nuage_nj:
        dist_nj.append([])
        for arr2 in nuage_nj:
            dist_nj[-1].append(sum((arr1[1]-arr2[1])*(arr1[1]-arr2[1])/centre_nj))

    # --- DataFrames des distances (AVEC LABELS) ---
    dist_ni_df = pd.DataFrame(dist_ni, index=rows_pl, columns=rows_pl)
    dist_nj_df = pd.DataFrame(dist_nj, index=cols_pc, columns=cols_pc)
    
    # --- Affichage des distances N(I) et N(J) ---
    st.header("Tableau de distance N(I)")
    st.dataframe(dist_ni_df)
    st.pyplot(bar_distance(dist_ni_df, "Distance N(I)"))
    st.pyplot(graph_dist_tab(dist_ni_df, "Distance N(I)"))

    st.header("Tableau de distance N(J)")
    st.dataframe(dist_nj_df)
    st.pyplot(bar_distance(dist_nj_df, "Distance N(J)"))
    st.pyplot(graph_dist_tab(dist_nj_df, "Distance N(J)"))

    #---------------- Calcul de dissimilarité (sur codage complet) --------
    taille_c = len(t_codage)
    t_distance = np.zeros(shape=(taille_c, taille_c))

    i = 0
    j = 0
    for p1 in t_codage:
        j = 0
        for p2 in t_codage:
            t_distance[i][j] = sum((abs(p1 - p2)))/ len(p1)
            j+=1
        i+=1

    t_distance_df = pd.DataFrame(t_distance, index=P_ind, columns=P_ind)
    
    st.header("Tableau de distance (dissimilarité)")
    st.dataframe(t_distance_df)
    st.pyplot(graph_dist_tab(t_distance_df, "Distance (dissimilarité)"))
    
    st.header("Tableau de distance (similarité)")
    t_similarite_df = 1 - t_distance_df
    st.dataframe(t_similarite_df)
    st.pyplot(graph_dist_tab(t_similarite_df, "Distance (similarité)"))