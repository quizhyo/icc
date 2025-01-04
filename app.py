import streamlit as st
import os
import time
import pandas as pd
from datetime import datetime
from streamlit_lottie import st_lottie
from dotenv import load_dotenv
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI  # Sử dụng đúng mô-đun từ langchain

# Import data analysis components
from src.utils.session import add_to_history
from src.data_analysis.prediction_model import prediction_model_pipeline
from src.data_analysis.cluster_model import cluster_model_pipeline
from src.data_analysis.regression_model import regression_model_pipeline
from src.data_analysis.visualization import data_visualization
from src.data_analysis.util import (
    load_lottie, 
    stream_data, 
    welcome_message, 
    introduction_message
)

# Import legal analysis components
from src.legal_analysis.agent import LegalAgentTeam
from src.legal_analysis.processor import DocumentProcessor

# Import utilities
from src.utils.session import init_session_state, validate_api_keys, init_qdrant

# Load environment variables
load_dotenv()

# Set up the Streamlit page configuration
st.set_page_config(
    layout="wide",
    page_title="AI-Powered Document Analysis",
    page_icon="🎓",
    initial_sidebar_state="expanded"
)


def display_history():
    """Display analysis history from session state"""
    if "history" in st.session_state and st.session_state.history:
        for entry in reversed(st.session_state.history):
            with st.expander(f"{entry['timestamp']} - {entry['file']}"):
                if entry['analysis_type']:
                    st.write(f"Analysis Type: {entry['analysis_type']}")
                if entry['query']:
                    st.write(f"Query: {entry['query']}")
                if entry['response']:
                    st.write(f"Response: {entry['response']}")
    else:
        st.write("No analysis history available.")


def init_openai():
    """Initialize OpenAI API configuration"""
    openai_key = os.getenv('OPENAI_API_KEY', '')
    if openai_key:
        st.session_state.openai_api_key = openai_key
        os.environ['OPENAI_API_KEY'] = openai_key
        return True
    else:
        st.sidebar.header("🔑 OpenAI API Configuration")
        openai_key = st.sidebar.text_input(
            "OpenAI API Key", 
            type="password",
            help="Enter your OpenAI API key"
        )
        if openai_key:
            st.session_state.openai_api_key = openai_key
            os.environ['OPENAI_API_KEY'] = openai_key
            return True
    return False

def setup_qdrant():
    """Setup Qdrant configuration for legal analysis"""
    qdrant_key = os.getenv('QDRANT_API_KEY', '')
    qdrant_url = os.getenv('QDRANT_URL', '')
    
    if qdrant_key and qdrant_url:
        st.session_state.qdrant_api_key = qdrant_key
        st.session_state.qdrant_url = qdrant_url
        try:
            st.session_state.vector_db = init_qdrant()
            return True
        except Exception as e:
            st.sidebar.error(f"Failed to connect to Qdrant: {str(e)}")
            return False
    else:
        st.sidebar.subheader("Qdrant Configuration")
        qdrant_key = st.sidebar.text_input(
            "Qdrant API Key",
            type="password",
            help="Enter your Qdrant API key"
        )
        qdrant_url = st.sidebar.text_input(
            "Qdrant URL",
            help="Enter your Qdrant URL"
        )
        
        if qdrant_key and qdrant_url:
            st.session_state.qdrant_api_key = qdrant_key
            st.session_state.qdrant_url = qdrant_url
            try:
                st.session_state.vector_db = init_qdrant()
                return True
            except Exception as e:
                st.sidebar.error(f"Failed to connect to Qdrant: {str(e)}")
                return False
    return False

def process_legal_document(uploaded_file):
    """Process legal document analysis"""
    success_placeholder = st.empty()
    success_placeholder.success("PDF file uploaded successfully!", icon="✅")
    time.sleep(2)
    success_placeholder.empty()
    
    if not hasattr(st.session_state, 'openai_api_key'):
        st.error("Please configure OpenAI API key first!")
        return
    
    if setup_qdrant():
        try:
            with st.spinner("Processing document..."):
                processor = DocumentProcessor(
                    vector_db=st.session_state.vector_db,
                    api_key=st.session_state.openai_api_key
                )
                
                st.session_state.knowledge_base = processor.process_document(uploaded_file)
                st.session_state.legal_team = LegalAgentTeam(st.session_state.knowledge_base)
            
            analysis_type = st.selectbox(
                "Select Analysis Type",
                [
                    "Contract Review",
                    "Legal Research",
                    "Risk Assessment",
                    "Compliance Check",
                    "Custom Query"
                ]
            )

            if analysis_type == "Custom Query":
                query = st.text_area("Enter your specific query:")
            else:
                query = None

            if st.button("Start Analysis"):
                with st.spinner("Analyzing document..."):
                    try:
                        response = st.session_state.legal_team.analyze(query, analysis_type)
                        
                        # Add to history
                        add_to_history(
                            entry_type="Legal Analysis",
                            file_name=uploaded_file.name,
                            analysis_type=analysis_type,
                            query=query,
                            response=response.content
                        )
                        
                        # Display results
                        tabs = st.tabs(["Analysis", "Key Points", "Recommendations"])
                        with tabs[0]:
                            st.markdown("### Detailed Analysis")
                            st.markdown(response.content)
                        
                        with tabs[1]:
                            st.markdown("### Key Points")
                            key_points = st.session_state.legal_team.analyze(
                                f"Summarize the key points from: {response.content}",
                                analysis_type
                            )
                            st.markdown(key_points.content)
                        
                        with tabs[2]:
                            st.markdown("### Recommendations")
                            recommendations = st.session_state.legal_team.analyze(
                                f"Provide recommendations based on: {response.content}",
                                analysis_type
                            )
                            st.markdown(recommendations.content)
                    
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
                        
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")

def display_home_page():
    """Display the home page content"""
    # Main Title with Animation
    st.markdown("""
    <h1 style='text-align: center; margin-bottom: 2rem;'>
        Transform Your Documents with AI-Powered Insights
    </h1>
    """, unsafe_allow_html=True)
    
    # Welcome Message
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        
    if st.session_state.initialized:
        st.session_state.welcome_message = welcome_message()
        st.write(stream_data(st.session_state.welcome_message))
        time.sleep(0.5)
        st.write("[Learn more about our features >](https://github.com/yourusername/yourrepo)")
        st.session_state.initialized = False
    else:
        st.write(st.session_state.welcome_message)
        st.write("[Learn more about our features >](https://github.com/yourusername/yourrepo)")

    # Features Overview
    st.markdown("## 🎯 Key Features")
    
    # Load Lottie animations if not already loaded
    if 'lottie' not in st.session_state:
        st.session_state.lottie_url1, st.session_state.lottie_url2 = load_lottie()
        st.session_state.lottie = True

    # Display features with animations
    col1, col2 = st.columns([6, 4])
    with col1:
        st.header("AI-Powered Analysis")
        st.write(introduction_message()[0])
    with col2:
        if st.session_state.lottie:
            st_lottie(st.session_state.lottie_url1, height=280, key="animation1")

def main():
    """Main application function"""
    # Initialize session state
    init_session_state()
    
    # Sidebar navigation
    with st.sidebar:
        # Logo
        st.image("images/full_logo.png", width=300)
        
        # Navigation
        pages = {
            "Home Page": "🏠",
            "AI Data Analysis": "📊",
            "Legal Analysis": "⚖️",
            "History": "📚"
        }
        
        for page in pages:
            button_style = "primary" if st.session_state.current_page == page else "secondary"
            if st.button(
                f"{pages[page]} {page}",
                key=f"nav_{page}",
                use_container_width=True,
                type=button_style
            ):
                st.session_state.current_page = page
                st.rerun()

    # Initialize OpenAI API
    if not init_openai():
        st.info("👈 Please enter your OpenAI API key to begin.")
        return

    # Main content
    if st.session_state.current_page == "Home Page":
        display_home_page()
    
    elif st.session_state.current_page == "AI Data Analysis":
        st.header("AI Data Analysis 📊")
        
        # Model and analysis type selection
        col1, col2 = st.columns(2)
        with col1:
            selected_model = st.selectbox(
                'Select OpenAI Model',
                ('GPT-4-Turbo', 'GPT-3.5-Turbo')
            )
        with col2:
            analysis_type = st.selectbox(
                'Select Analysis Type',
                ['Predictive Classification', 'Clustering Model', 
                 'Regression Model', 'Data Visualization']
            )
        
        uploaded_file = st.file_uploader("Upload your data file", 
                                       type=['csv', 'xlsx', 'xls', 'json'])
        
        # Convert uploaded file to DataFrame and store in session state
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    st.session_state.DF_uploaded = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                    st.session_state.DF_uploaded = pd.read_excel(uploaded_file)
                else:
                    st.error("Unsupported file format")
                    return
                if df.empty:
                    st.error("The uploaded file is empty")
                    return
                st.session_state.DF_uploaded = df
                
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                st.session_state.DF_uploaded = None
                st.session_state.is_file_empty = True

        # Proceed Button
        is_proceed_enabled = uploaded_file is not None and st.session_state.openai_api_key != ""
        
        if 'button_clicked' not in st.session_state:
            st.session_state.button_clicked = False
            
        if st.button('Start Analysis', 
                     disabled=(not is_proceed_enabled) or st.session_state.button_clicked, 
                     type="primary"):
            st.session_state.button_clicked = True
            
        if "is_file_empty" in st.session_state and st.session_state.is_file_empty:
            st.caption('Your data file is empty!')

        # Start Analysis
        if st.session_state.button_clicked:
            gpt_model = 4 if selected_model == 'GPT-4-Turbo' else 3.5
            with st.container():
                if "DF_uploaded" not in st.session_state or st.session_state.DF_uploaded is None:
                    st.error("File is empty!")
                else:
                    if analysis_type == "Predictive Classification":
                        prediction_model_pipeline(st.session_state.DF_uploaded, 
                                               st.session_state.openai_api_key, 
                                               gpt_model)
                    elif analysis_type == "Clustering Model":
                        cluster_model_pipeline(st.session_state.DF_uploaded, 
                                            st.session_state.openai_api_key, 
                                            gpt_model)
                    elif analysis_type == "Regression Model":
                        regression_model_pipeline(st.session_state.DF_uploaded, 
                                               st.session_state.openai_api_key, 
                                               gpt_model)
                    elif analysis_type == "Data Visualization":
                        data_visualization(st.session_state.DF_uploaded)

    
    elif st.session_state.current_page == "Legal Analysis":
        st.header("Legal Document Analysis ⚖️")
        uploaded_file = st.file_uploader("Upload PDF Document", type=['pdf'])
        if uploaded_file:
            process_legal_document(uploaded_file)
    
    elif st.session_state.current_page == "History":
        display_history()

    # Footer
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: #666666;'>
            <p><strong>Disclaimer</strong></p>
            <p>This is an AI-powered tool. Results should be reviewed by professionals for critical matters.</p>
            <p>© 2024 All rights reserved.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    if st.session_state.current_page == "History":
        display_history()  # Now this function is defined

if __name__ == "__main__":
    main()