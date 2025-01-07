import streamlit as st
from phi.vectordb.qdrant import Qdrant
from typing import Optional, Union
import pandas as pd
from datetime import datetime

def init_session_state():
    """Initialize all session state variables"""
    # API Keys and Connections
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = None
    if 'qdrant_api_key' not in st.session_state:
        st.session_state.qdrant_api_key = None
    if 'qdrant_url' not in st.session_state:
        st.session_state.qdrant_url = None
    if 'vector_db' not in st.session_state:
        st.session_state.vector_db = None

    # Legal Analysis State
    if 'legal_team' not in st.session_state:
        st.session_state.legal_team = None
    if 'knowledge_base' not in st.session_state:
        st.session_state.knowledge_base = None

    # Data Analysis State
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'df_origin' not in st.session_state:
        st.session_state.df_origin = None
    if 'data_agent' not in st.session_state:
        st.session_state.data_agent = None
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'analysis_mode' not in st.session_state:
        st.session_state.analysis_mode = None
    if 'target_Y' not in st.session_state:
        st.session_state.target_Y = None
    if 'target_selected' not in st.session_state:
        st.session_state.target_selected = False
    if 'contain_null' not in st.session_state:
        st.session_state.contain_null = None
    if 'all_numeric' not in st.session_state:
        st.session_state.all_numeric = None
    if 'to_perform_pca' not in st.session_state:
        st.session_state.to_perform_pca = None

    # Model State
    if 'model_list' not in st.session_state:
        st.session_state.model_list = None
    if 'model1' not in st.session_state:
        st.session_state.model1 = None
    if 'model2' not in st.session_state:
        st.session_state.model2 = None
    if 'model3' not in st.session_state:
        st.session_state.model3 = None
    
    # Data Processing State
    if 'filled_df' not in st.session_state:
        st.session_state.filled_df = None
    if 'encoded_df' not in st.session_state:
        st.session_state.encoded_df = None
    if 'df_cleaned1' not in st.session_state:
        st.session_state.df_cleaned1 = None
    if 'df_cleaned2' not in st.session_state:
        st.session_state.df_cleaned2 = None
    if 'df_pca' not in st.session_state:
        st.session_state.df_pca = None

    # UI State
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Home Page"
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
    if 'start_training' not in st.session_state:
        st.session_state.start_training = False
    if 'button_clicked' not in st.session_state:
        st.session_state.button_clicked = False

def init_qdrant() -> Optional[Qdrant]:
    """Initialize Qdrant vector database connection"""
    try:
        if not st.session_state.qdrant_api_key:
            raise ValueError("Qdrant API key not provided")
        if not st.session_state.qdrant_url:
            raise ValueError("Qdrant URL not provided")

        vector_db = Qdrant(
            collection="legal_knowledge",
            url=st.session_state.qdrant_url,
            api_key=st.session_state.qdrant_api_key,
            https=True,
            timeout=None,
            distance="cosine"
        )
        return vector_db
    except Exception as e:
        st.error(f"Failed to connect to Qdrant: {str(e)}")
        return None

def handle_file_upload(uploaded_file) -> Union[pd.DataFrame, None]:
    """Handle file upload based on file type"""
    if uploaded_file is None:
        return None

    file_type = uploaded_file.name.split('.')[-1].lower()
    try:
        if file_type == 'csv':
            st.session_state.analysis_mode = 'data'
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            return df
        elif file_type == 'pdf':
            st.session_state.analysis_mode = 'legal'
            return None
        else:
            st.error(f"Unsupported file type: {file_type}")
            return None
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def validate_api_keys() -> bool:
    """Validate required API keys are present"""
    if not st.session_state.openai_api_key:
        st.warning("Please provide your OpenAI API key.")
        return False
    
    if st.session_state.analysis_mode == 'legal':
        if not st.session_state.qdrant_api_key or not st.session_state.qdrant_url:
            st.warning("Please provide Qdrant credentials for legal analysis.")
            return False
    
    return True

def clear_session():
    """Clear session state"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_session_state()

def add_to_history(entry_type: str, file_name: str, analysis_type: str = None, 
                  query: str = None, response: str = None):
    """Add an entry to the analysis history"""
    # Initialize history if it doesn't exist
    if "history" not in st.session_state:
        st.session_state.history = []
        
    entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'type': entry_type,
        'file': file_name,
        'analysis_type': analysis_type,
        'query': query,
        'response': response
    }
    st.session_state.history.append(entry)

def display_history():
    """Display analysis history"""
    # Check if history exists in session state
    if "history" not in st.session_state or not st.session_state.history:
        st.info("No analysis history yet.")
        return

    for entry in reversed(st.session_state.history):
        with st.expander(f"{entry['timestamp']} - {entry['type']}"):
            st.write(f"**Type:** {entry['type']}")
            st.write(f"**File:** {entry['file']}")
            if entry['analysis_type']:
                st.write(f"**Analysis Type:** {entry['analysis_type']}")
            if entry['query']:
                st.write(f"**Query:** {entry['query']}")
            if entry['response']:
                st.write("**Response:**")
                st.markdown(entry['response'])

def reset_model_state():
    """Reset model-related session state variables"""
    model_vars = ['model_list', 'model1', 'model2', 'model3', 
                  'start_training', 'button_clicked']
    for var in model_vars:
        if var in st.session_state:
            del st.session_state[var]

def reset_data_state():
    """Reset data processing session state variables"""
    data_vars = ['df', 'df_origin', 'filled_df', 'encoded_df', 
                 'df_cleaned1', 'df_cleaned2', 'df_pca']
    for var in data_vars:
        if var in st.session_state:
            del st.session_state[var]