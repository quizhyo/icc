import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import pygwalker as pyg
import io
from typing import Optional

def execute_plot_code(code: str, df: pd.DataFrame, fig_size: tuple = (10, 6)) -> Optional[plt.Figure]:
    """Execute plot code safely and return matplotlib figure"""
    try:
        plt.figure(figsize=fig_size)
        local_vars = {"plt": plt, "df": df}
        compiled_code = compile(code, "<string>", "exec")
        exec(compiled_code, globals(), local_vars)
        return plt.gcf()
    except Exception as e:
        st.error(f"Error executing plot code: {e}")
        return None

def get_pygwalker_html(df: pd.DataFrame) -> str:
    """Generate interactive visualization using PyGWalker"""
    return pyg.to_html(df)

def display_data_preview(df: pd.DataFrame):
    """Display data preview and statistics in Streamlit"""
    st.write("### Data Preview:")
    st.write(df.head())
    
    # Display basic statistics
    st.write("### Data Statistics:")
    st.write(df.describe())
    
    # Display column info
    st.write("### Column Information:")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())
    
    # Display missing values info
    st.write("### Missing Values:")
    missing_data = df.isnull().sum()
    if missing_data.any():
        st.write(missing_data[missing_data > 0])
    else:
        st.write("No missing values found!")

def create_basic_visualizations(df: pd.DataFrame):
    """Create basic visualizations for dataset overview"""
    # Numerical columns distribution
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if not num_cols.empty:
        st.write("### Numerical Columns Distribution:")
        for col in num_cols:
            fig, ax = plt.subplots(figsize=(10, 6))
            df[col].hist(ax=ax)
            plt.title(f'Distribution of {col}')
            st.pyplot(fig)
            plt.close()
    
    # Categorical columns distribution
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if not cat_cols.empty:
        st.write("### Categorical Columns Distribution:")
        for col in cat_cols:
            if df[col].nunique() <= 10:  # Only for columns with reasonable number of categories
                fig, ax = plt.subplots(figsize=(10, 6))
                df[col].value_counts().plot(kind='bar', ax=ax)
                plt.title(f'Distribution of {col}')
                plt.xticks(rotation=45)
                st.pyplot(fig)
                plt.close()

def create_correlation_matrix(df: pd.DataFrame):
    """Create and display correlation matrix for numerical columns"""
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(num_cols) > 1:
        st.write("### Correlation Matrix:")
        corr_matrix = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
        plt.colorbar()
        plt.xticks(range(len(num_cols)), num_cols, rotation=45)
        plt.yticks(range(len(num_cols)), num_cols)
        st.pyplot(fig)
        plt.close()