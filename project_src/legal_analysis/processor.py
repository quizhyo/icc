from phi.knowledge.pdf import PDFKnowledgeBase, PDFReader
from phi.vectordb.qdrant import Qdrant
from phi.embedder.openai import OpenAIEmbedder
import tempfile
import os
import streamlit as st

class DocumentProcessor:
    def __init__(self, vector_db: Qdrant, api_key: str):
        """
        Initialize DocumentProcessor
        Args:
            vector_db (Qdrant): Initialized Qdrant vector database
            api_key (str): OpenAI API key
        """
        self.vector_db = vector_db
        self.api_key = api_key

    def process_document(self, uploaded_file) -> PDFKnowledgeBase:
        """
        Process PDF document and create knowledge base
        Args:
            uploaded_file: Streamlit uploaded file object
        Returns:
            PDFKnowledgeBase: Processed knowledge base
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Save uploaded file to temporary directory
                temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Create embeddings and knowledge base
                embedder = OpenAIEmbedder(
                    model="text-embedding-3-small",
                    api_key=self.api_key
                )
                
                # Initialize knowledge base with settings
                knowledge_base = PDFKnowledgeBase(
                    path=temp_dir,
                    vector_db=self.vector_db,
                    reader=PDFReader(chunk=True),
                    embedder=embedder,
                    recreate_vector_db=True
                )

                # Load and process the document
                with st.spinner("Processing document..."):
                    knowledge_base.load()
                    
                st.success("✅ Document processed successfully!")
                return knowledge_base

            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
                raise

    @staticmethod
    def validate_pdf(uploaded_file) -> bool:
        """
        Validate if the uploaded file is a valid PDF
        Args:
            uploaded_file: Streamlit uploaded file object
        Returns:
            bool: True if valid, False otherwise
        """
        if uploaded_file is None:
            return False
            
        # Check file type
        file_type = uploaded_file.name.split('.')[-1].lower()
        if file_type != 'pdf':
            st.error("Please upload a PDF file.")
            return False

        # Check file size (max 10MB)
        if uploaded_file.size > 10 * 1024 * 1024:
            st.error("File size should be less than 10MB.")
            return False

        return True

    def get_document_info(self, knowledge_base: PDFKnowledgeBase) -> dict:
        """
        Get information about the processed document
        Args:
            knowledge_base: Processed knowledge base
        Returns:
            dict: Document information
        """
        try:
            # Get basic document info
            doc_info = {
                "chunks": len(knowledge_base.chunks),
                "embeddings": knowledge_base.embeddings.shape if knowledge_base.embeddings is not None else None,
                "vector_db_status": "Connected" if self.vector_db else "Not Connected"
            }

            # Get document metadata if available
            if hasattr(knowledge_base, 'metadata'):
                doc_info.update(knowledge_base.metadata)

            return doc_info

        except Exception as e:
            st.warning(f"Could not retrieve complete document info: {str(e)}")
            return {}