def add_to_history(
    entry_type: str,
    filename: str,
    model: str,
    analysis_type: str
) -> None:
    """Add entry to analysis history"""
    if 'history' not in st.session_state:
        st.session_state.history = []
        
    entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'type': entry_type,
        'filename': filename,
        'model': model,
        'analysis_type': analysis_type
    }
    
    st.session_state.history.append(entry)