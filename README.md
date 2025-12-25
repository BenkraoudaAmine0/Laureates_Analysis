# Analyse des Lauréats - Data Analysis Tool

This repository contains a Streamlit application designed for the analysis of categorical data, with a specific focus on analyzing laureates' data (e.g., Nobel Prize winners). The tool provides a comprehensive suite of statistical methods and visualizations to explore relationships between variables and individuals.

## Features

The application (`TPandFinal.py`) includes the following key functionalities:

### 1. Data Loading & Preprocessing
*   **Excel Upload**: Support for uploading `.xlsx` files containing raw data.
*   **Sampling**: Option to work on a random sample of the dataset for faster processing.
*   **Cleaning**: Automated options to remove missing values (NaN) and ensure correct data types.

### 2. Descriptive Statistcs & Contingency Analysis
*   **Variable Selection**: Interactively select row and column variables for analysis.
*   **Contingency Tables**: Generates contingency tables with dynamic filtering based on minimum observations per row/column.

### 3. Advanced Statistical Analysis (MCA Logic)
*   **Variable Typing**: Define variables as **Nominal** or **Ordinal**.
    *   *Ordinal Variables*: Interactive drag-and-drop plotting to reorder modalities.
*   **Coding & Burt Tables**: Automatic generation of the Complete Disjunctive Table (Tableau de Codage) and Burt Table.
*   **Profile Analysis**:
    *   Calculation of Row and Column profiles from frequency tables.
    *   Weighted margins and distribution analysis.

### 4. Distance & Similarity Metrics
*   **Chi-Square Distances**:
    *   Calculation of distances between Row profiles (N(I)) and Column profiles (N(J)).
    *   **Visualizations**: Bar charts, Heatmaps, and 1D Distance Scale plots to visualize the proximity between categories.
*   **Global Dissimilarity/Similarity**:
    *   Computation of inter-individual distances using normalized Cityblock (Manhattan) distance.
    *   Dissimilarity and Similarity matrices.

### 5. Geospatial Visualization
*   **Interactive Map**: A Plotly Choropleth map focusing on Europe to visualize the distribution of laureates by "Born country".

## Requirements

To run this application, you need Python installed along with the following libraries:

*   streamlit
*   pandas
*   matplotlib
*   numpy
*   seaborn
*   plotly
*   scipy
*   openpyxl

## Installation

1.  Clone this repository.
2.  Install the required dependencies:

```bash
pip install streamlit pandas matplotlib numpy seaborn plotly scipy openpyxl
```

## Usage

Run the Streamlit application using the following command in your terminal:

```bash
streamlit run TPandFinal.py
```

Comparison and analysis results will be displayed directly in your web browser.

## Project Origin
This project appears to be part of an academic assignment (TP - Travaux Pratiques) for analysing data analysis algorithms and visualizations, specifically focusing on Factorial Correspondence analysis concepts (distances, profiles, specificities).
