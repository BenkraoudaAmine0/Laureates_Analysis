import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import seaborn as sns
from io import BytesIO

# --- Plotting Functions (Improved from tp0fin.py) ---

def plot_heatmap(dist_df: pd.DataFrame, title: str):
    """
    Generates and returns a heatmap for the given distance matrix.
    Uses seaborn for a clearer visualization.
    """
    # Dynamic figure size based on matrix dimensions
    width = max(10, len(dist_df.columns) * 0.5)
    height = max(8, len(dist_df.index) * 0.5)
    
    fig, ax = plt.subplots(figsize=(width, height))
    
    sns.heatmap(
        dist_df,
        annot=True,        # Show the distance values
        fmt=".3f",         # Format values to 3 decimal places
        linewidths=.5,
        ax=ax,
        cmap="viridis_r"   # Use a "reverse" colormap (low = good)
    )
    
    ax.set_title(f"Heatmap of {title}", fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    return fig

# --- Core Analysis Functions ---

def calculate_chi2_distances(profiles_df: pd.DataFrame, margins: pd.Series):
    """
    Calculates the Chi-squared distance matrix between the rows of a profiles DataFrame.
    
    Args:
        profiles_df: DataFrame of profiles (e.g., row profiles).
        margins: Series of margins (e.g., column margins) for weighting.
    """
    n_rows = len(profiles_df)
    dist_matrix = pd.DataFrame(
        np.zeros((n_rows, n_rows)),
        index=profiles_df.index,
        columns=profiles_df.index
    )
    
    # Ensure margins are not zero to avoid division errors
    safe_margins = margins.copy()
    safe_margins[safe_margins == 0] = 1e-9 # Avoid division by zero
    
    for i in range(n_rows):
        for j in range(i, n_rows):  # Only compute upper triangle
            if i == j:
                distance = 0.0
            else:
                prof_i = profiles_df.iloc[i]
                prof_j = profiles_df.iloc[j]
                
                # Chi-squared distance formula
                distance = (((prof_i - prof_j) ** 2) / safe_margins).sum()
                
            dist_matrix.iloc[i, j] = distance
            dist_matrix.iloc[j, i] = distance # Matrix is symmetric
            
    return dist_matrix

def run_correspondence_analysis(contingency_table: pd.DataFrame):
    """
    Performs the full analysis (profiles, distances) on a contingency table.
    """
    st.header("1. Contingence Table", divider="blue")
    st.dataframe(contingency_table)
    
    # --- Basic Calculations (from tp0fin.py) ---
    total_g = contingency_table.sum().sum()
    if total_g == 0:
        st.error("The contingency table is empty or has a total sum of zero.")
        return

    # Row and column sums (margins)
    row_sums = contingency_table.sum(axis=1)
    col_sums = contingency_table.sum(axis=0)

    # --- Profile Calculations ---
    
    # Frequency Table (t_fi)
    st.header("2. Frequency Table (t_fi)", divider="blue")
    st.caption("Each cell divided by the grand total.")
    t_fi = contingency_table / total_g
    st.dataframe(t_fi.style.format("{:.4f}"))

    # Row Profiles (t_pl)
    st.header("3. Row Profiles (t_pl)", divider="blue")
    st.caption("Each row divided by its total (row sum). Rows sum to 1.")
    # Avoid division by zero for empty rows
    safe_row_sums = row_sums.replace(0, 1e-9)
    t_pl = contingency_table.div(safe_row_sums, axis=0)
    st.dataframe(t_pl.style.format("{:.4f}"))

    # Column Profiles (t_pc)
    st.header("4. Column Profiles (t_pc)", divider="blue")
    st.caption("Each column divided by its total (column sum). Columns sum to 1.")
    # Avoid division by zero for empty columns
    safe_col_sums = col_sums.replace(0, 1e-9)
    t_pc = contingency_table.div(safe_col_sums, axis=1)
    st.dataframe(t_pc.style.format("{:.4f}"))

    # --- Distance Calculations ---
    
    # Margins (as frequencies)
    col_margins_fi = col_sums / total_g
    row_margins_fi = row_sums / total_g

    # Distance between Row Profiles
    st.header("5. Distances Between Row Profiles (Chi-Squared)", divider="blue")
    st.caption("Measures how different the row profiles are from each other.")
    try:
        dist_rows = calculate_chi2_distances(t_pl, col_margins_fi)
        st.dataframe(dist_rows.style.format("{:.4f}"))
        st.pyplot(plot_heatmap(dist_rows, "Row Profile Distances"))
    except Exception as e:
        st.error(f"Could not calculate row distances: {e}")

    # Distance between Column Profiles
    st.header("6. Distances Between Column Profiles (Chi-Squared)", divider="blue")
    st.caption("Measures how different the column profiles are from each other.")
    try:
        # We calculate distances on the *transposed* profiles (t_pc.T)
        # and weight by the *row* margins.
        dist_cols = calculate_chi2_distances(t_pc.T, row_margins_fi)
        st.dataframe(dist_cols.style.format("{:.4f}"))
        st.pyplot(plot_heatmap(dist_cols, "Column Profile Distances"))
    except Exception as e:
        st.error(f"Could not calculate column distances: {e}")


# --- Streamlit App Main ---

def main():
    st.set_page_config(layout="wide")

    # File uploader for Excel files
    uploaded_file = st.file_uploader(
        "Choose an Excel file",
        type=["xlsx", "xls"]
    )

    if uploaded_file is not None:
        try:
            # Read the Excel file. 
            # 'engine='openpyxl' is needed for .xlsx files
            df = pd.read_excel(uploaded_file, engine='openpyxl')
            
            st.success("File uploaded and read successfully!")
            st.dataframe(df, use_container_width=True)

            # Find categorical columns for user selection
            # These are 'object' (text) or 'category' types
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if not cat_cols or len(cat_cols) < 2:
                st.warning("Could not find at least two categorical (text) columns to analyze.")
                return

            # User selection for row and column variables
            st.subheader("Create Contingency Table")
            col1, col2 = st.columns(2)
            with col1:
                row_var = st.selectbox("Select Row Variable:", cat_cols, index=0)
            with col2:
                # Pre-select a different column
                default_col_index = 1 if len(cat_cols) > 1 else 0
                col_var = st.selectbox("Select Column Variable:", cat_cols, index=default_col_index)
            
            if row_var == col_var:
                st.warning("Please select two different variables.")
            else:
                if st.button(f"Analyze: {row_var} vs. {col_var}", type="primary"):
                    try:
                        # Create the contingency table
                        contingency_table = pd.crosstab(df[row_var], df[col_var])
                        
                        # --- Optional: Filter small categories ---
                        st.sidebar.subheader("Analysis Filters")
                        min_row_sum = st.sidebar.slider("Min observations per row", 0, 100, 3)
                        min_col_sum = st.sidebar.slider("Min observations per col", 0, 100, 3)

                        # Filter table
                        filtered_table = contingency_table.loc[contingency_table.sum(axis=1) >= min_row_sum]
                        filtered_table = filtered_table.loc[:, filtered_table.sum(axis=0) >= min_col_sum]

                        if filtered_table.empty:
                            st.error("Filtered table is empty with current settings. Try lowering the filters.")
                        else:
                            # Run the full analysis
                            run_correspondence_analysis(filtered_table)

                    except Exception as e:
                        st.error(f"An error occurred during analysis: {e}")

        except Exception as e:
            st.error(f"Error reading Excel file: {e}")
            st.warning("Please ensure the file is a valid .xlsx or .xls file.")

if __name__ == "__main__":
    main()