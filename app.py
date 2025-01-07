import streamlit as st
import os
import time
import pandas as pd
from datetime import datetime
from streamlit_lottie import st_lottie
from dotenv import load_dotenv
from langchain.agents import AgentType
from langchain_community.chat_models import ChatOpenAI
from PIL import Image

# Import data analysis components
from project_src.utils.session import init_session_state, validate_api_keys, init_qdrant
from project_src.legal_analysis.agent import LegalAgentTeam
from project_src.legal_analysis.processor import DocumentProcessor
from project_src.data_analysis.prediction_model import prediction_model_pipeline
from project_src.data_analysis.src.util import read_file_from_streamlit
from project_src.data_analysis.cluster_model import cluster_model_pipeline
from project_src.data_analysis.regression_model import regression_model_pipeline
from project_src.data_analysis.visualization import data_visualization
from project_src.data_analysis.data_utils import (
    load_lottie, 
    stream_data, 
    welcome_message, 
    introduction_message
)
from project_src.data_analysis_v1.agent import DataAnalysisAgent
from project_src.data_analysis_v1.visualizer import display_data_preview, create_basic_visualizations, create_correlation_matrix
# Load environment variables
load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')

im = Image.open("images/only_logo.png")
# Set up the Streamlit page configuration

# Set up the Streamlit page configuration
st.set_page_config(
    layout="wide",
    page_title="ICC AI Agent - R&D Project",
    page_icon=im,
    initial_sidebar_state="expanded"
)

def setup_qdrant():
    """Setup Qdrant configuration"""
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
    return False

def process_pdf(uploaded_file):
    """Process PDF file for legal analysis"""
    success_placeholder = st.empty()
    success_placeholder.success("PDF file uploaded successfully!", icon="✅")
    time.sleep(2)
    success_placeholder.empty()
    
    if not API_KEY:
        st.error("OpenAI API key not found in environment variables.")
        return
    
    if setup_qdrant():
        try:
            with st.spinner("Processing document..."):
                processor = DocumentProcessor(
                    vector_db=st.session_state.vector_db,
                    api_key=API_KEY
                )
                
                st.session_state.knowledge_base = processor.process_document(uploaded_file)
                st.session_state.legal_team = LegalAgentTeam(st.session_state.knowledge_base)
            
            # Analysis Options
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

            if st.button("Start Analysis", key="legal_analysis"):
                with st.spinner("Analyzing document..."):
                    try:
                        # Get initial analysis
                        response = st.session_state.legal_team.analyze(query, analysis_type)
                        
                        # Add to history
                        add_to_history(
                            entry_type="Legal Analysis",
                            file_name=uploaded_file.name,
                            analysis_type=analysis_type,
                            query=query,
                            response=response.content
                        )
                        
                        # Display results in tabs
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
def process_csv(uploaded_file):
    """Process CSV file for data analysis"""
    success_placeholder = st.empty()
    success_placeholder.success("CSV file uploaded successfully!", icon="✅")
    time.sleep(5)
    success_placeholder.empty()
    
    df = pd.read_csv(uploaded_file)
    display_data_preview(df)
    
    query = st.text_input("Enter your analysis query:")
    if st.button("Analyze"):
        with st.spinner("Processing..."):
            llm = ChatOpenAI(temperature=0)
            agent = DataAnalysisAgent(
                df=df,
                llm=llm,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=False,
                return_intermediate_steps=True
            )

            try:
                result = agent.analyze(query)
                if isinstance(result, dict) and 'output' in result:
                    result = result['output']
                
                # Add to history
                add_to_history(
                    file_name=uploaded_file.name,
                    file_type="CSV",
                    analysis_type="Data Analysis",
                    query=query,
                    response=result
                )
                
                tabs = st.tabs(["Analysis Results", "Visualizations"])
                
                with tabs[0]:
                    st.markdown("### Analysis Results")
                    st.write(result)
                
                with tabs[1]:
                    st.markdown("### Data Visualizations")
                    create_basic_visualizations(df)
                    create_correlation_matrix(df)
                
            except Exception as e:
                error_placeholder = st.empty()
                error_placeholder.error(f"Analysis error: {str(e)}")
                time.sleep(5)
                error_placeholder.empty()
def display_history():
    """Display analysis history from session state"""
    if "history" in st.session_state and st.session_state.history:
        for entry in reversed(st.session_state.history):
            with st.expander(f"{entry['timestamp']} - {entry['file']}"):
                if entry.get('analysis_type'):
                    st.write(f"Analysis Type: {entry['analysis_type']}")
                if entry.get('query'):
                    st.write(f"Query: {entry['query']}")
                if entry.get('response'):
                    st.write(f"Response: {entry['response']}")
    else:
        st.write("No analysis history available.")

def display_home_page():
    """Display the home page content"""
    st.subheader("Hello there 👋")
    st.title("Welcome to AI-Powered Analysis!")
    
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        
    if st.session_state.initialized:
        st.session_state.welcome_message = welcome_message()
        st.write(stream_data(st.session_state.welcome_message))
        time.sleep(0.5)
        st.write("[Learn more about our features >](hung.dang@intersnack.com.vn)")
        st.session_state.initialized = False
    else:
        st.write(st.session_state.welcome_message)
        st.write("[Learn more about our features >](hung.dang@intersnack.com.vn)")

    # Introduction section
    st.divider()
    if 'lottie' not in st.session_state:
        st.session_state.lottie_url1, st.session_state.lottie_url2 = load_lottie()
        st.session_state.lottie = True

    left_column_r1, right_column_r1 = st.columns([6, 4])
    with left_column_r1:
        st.header("What can our AI Analysis do?")
        st.write(introduction_message()[0])
    with right_column_r1:
        if st.session_state.lottie:
            st_lottie(st.session_state.lottie_url1, height=280, key="animation1")

    left_column_r2, _, right_column_r2 = st.columns([6, 1, 5])
    with left_column_r2:
        if st.session_state.lottie:
            st_lottie(st.session_state.lottie_url2, height=200, key="animation2")
    with right_column_r2:
        st.header("Simple to Use")
        st.write(introduction_message()[1])

def add_to_history(
    entry_type: str = "Data Analysis",
    filename: str = None,
    file_name: str = None,  # For backwards compatibility
    file_type: str = None,
    model: str = None,
    analysis_type: str = None,
    query: str = None,
    response: str = None
) -> None:
    """Add entry to analysis history"""
    if 'history' not in st.session_state:
        st.session_state.history = []
        
    entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'type': entry_type,
        'file': filename or file_name,  # Use either filename or file_name
        'file_type': file_type,
        'model': model,
        'analysis_type': analysis_type,
        'query': query,
        'response': response
    }
    
    # Remove None values from the entry
    entry = {k: v for k, v in entry.items() if v is not None}
    
    st.session_state.history.append(entry)

def display_data_analysis():
    """Display AI Data Analysis page"""
    st.header("AI Data Analysis 📊")
    
    # Create two columns for file upload and model selection
    left_column, right_column = st.columns([6, 4])
    
    with left_column:
        uploaded_file = st.file_uploader(
            "Upload your data file",
            type=['csv', 'xlsx', 'xls', 'json'],
            help="Upload CSV or Excel file"
        )
        
        if uploaded_file:
            if uploaded_file.getvalue():
                uploaded_file.seek(0)
                try:
                    st.session_state.DF_uploaded = read_file_from_streamlit(uploaded_file)
                    st.session_state.is_file_empty = False
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
                    st.session_state.is_file_empty = True
                    return
            else:
                st.session_state.is_file_empty = True
                st.error("The uploaded file is empty")
                return
    
    with right_column:
        SELECTED_MODEL = st.selectbox(
            'Which OpenAI model do you want to use?',
            ('GPT-4-Turbo', 'GPT-3.5-Turbo')
        )
        
        MODE = st.selectbox(
            'Select proper data analysis mode',
            ['Predictive Classification', 'Clustering Model', 
             'Regression Model', 'Data Visualization']
        )
        
        st.write(f'Model selected: :green[{SELECTED_MODEL}]')
        st.write(f'Data analysis mode: :green[{MODE}]')

    # Initialize the 'button_clicked' state
    if 'button_clicked' not in st.session_state:
        st.session_state.button_clicked = False

    # Proceed Button    
    is_proceed_enabled = uploaded_file is not None and not st.session_state.get('is_file_empty', True)
    
    if st.button("Start Analysis", 
                 disabled=(not is_proceed_enabled) or st.session_state.button_clicked,
                 type="primary"):
        st.session_state.button_clicked = True
        
    if "is_file_empty" in st.session_state and st.session_state.is_file_empty:
        st.caption('Your data file is empty!')

    # Start Analysis
    if st.session_state.button_clicked:
        GPT_MODEL = 4 if SELECTED_MODEL == 'GPT-4-Turbo' else 3.5
        
        with st.container():
            if "DF_uploaded" not in st.session_state:
                st.error("File is empty!")
            else:
                with st.spinner("Analyzing data..."):
                    try:
                        ## Add to history using filename from session state
                        if hasattr(st.session_state, 'uploaded_filename'):
                            add_to_history(
                                entry_type="Data Analysis",
                                filename=st.session_state.uploaded_filename,
                                model=SELECTED_MODEL,
                                analysis_type=MODE
                            )

                        try:
                            if MODE == "Predictive Classification":
                                prediction_model_pipeline(st.session_state.DF_uploaded, API_KEY, GPT_MODEL)
                            elif MODE == "Clustering Model":
                                cluster_model_pipeline(st.session_state.DF_uploaded, API_KEY, GPT_MODEL) 
                            elif MODE == "Regression Model":
                                regression_model_pipeline(st.session_state.DF_uploaded, API_KEY, GPT_MODEL)
                            elif MODE == "Data Visualization":
                                data_visualization(st.session_state.DF_uploaded, API_KEY, GPT_MODEL)
                        except Exception as e:
                            st.error(f"Error during analysis: {str(e)}")
                        except Exception as e:
                            st.error(f"Error during analysis: {str(e)}")
                    except Exception as e:
                            st.error(f"Error during analysis: {str(e)}")

def main():
    """Main application function"""
    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Home Page"
    
    init_session_state()
    
    # Sidebar navigation
    with st.sidebar:
        st.image("images/full_logo.png", width=300)
        
        # Navigation
        pages = {
            "Home Page": "🏠",
            "AI Data Analysis V1": "🤖",
            "AI Data Analysis V2": "📊",
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

   

    # Main content
    if st.session_state.current_page == "Home Page":
        display_home_page()
    
    elif st.session_state.current_page == "AI Data Analysis V2":
        display_data_analysis()
    elif st.session_state.current_page == "AI Data Analysis V1":
        st.markdown("## AI Data Analysis 🤖")
        uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
        if uploaded_file:
            process_csv(uploaded_file)
    elif st.session_state.current_page == "Legal Analysis":
        st.header("Legal Document Analysis ⚖️")
        st.info("📄 Upload your legal documents in PDF format for AI-powered analysis")
        
        uploaded_file = st.file_uploader(
            "Upload PDF Document",
            type=['pdf'],
            key="legal_doc_upload"  # Changed from 'legal_analysis'
        )
        if uploaded_file:
            success_placeholder = st.empty()
            success_placeholder.success("PDF file uploaded successfully!", icon="✅")
            time.sleep(2)
            success_placeholder.empty()
            
            if not API_KEY:
                st.error("OpenAI API key not found in environment variables.")
                return
            
            if setup_qdrant():
                try:
                    with st.spinner("Processing document..."):
                        processor = DocumentProcessor(
                            vector_db=st.session_state.vector_db,
                            api_key=API_KEY
                        )
                        
                        st.session_state.knowledge_base = processor.process_document(uploaded_file)
                        st.session_state.legal_team = LegalAgentTeam(st.session_state.knowledge_base)
                    
                    # Analysis Options
                    analysis_type = st.selectbox(
                        "Select Analysis Type",
                        [
                            "Contract Review",
                            "Legal Research",
                            "Risk Assessment",
                            "Compliance Check",
                            "Custom Query"
                        ],
                        key="analysis_type_select"  # Added unique key
                    )

                    if analysis_type == "Custom Query":
                        query = st.text_area("Enter your specific query:", key="custom_query_input")  # Added unique key
                    else:
                        query = None

                    if st.button("Start Analysis", key="start_legal_analysis"):  # Changed from 'legal_analysis'
                        with st.spinner("Analyzing document..."):
                            try:
                                # Get initial analysis
                                response = st.session_state.legal_team.analyze(query, analysis_type)
                                
                                # Add to history
                                add_to_history(
                                    entry_type="Legal Analysis",
                                    file_name=uploaded_file.name,
                                    analysis_type=analysis_type,
                                    query=query,
                                    response=response.content
                                )
                                
                                # Display results in tabs
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
    
    elif st.session_state.current_page == "History":
        display_history()

    # Footer
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: #666666;'>
            <p><strong>Disclaimer</strong></p>
            <p>This is an AI-powered tool. Results should be reviewed by professionals.</p>
            <p>© 2024 All rights reserved.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()