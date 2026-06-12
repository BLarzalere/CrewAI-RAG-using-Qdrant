import streamlit as st
from torch_rag.main import run_from_gui
from PIL import Image

im = Image.open("src/torch_rag/new-logo.png")
st.set_page_config(page_title="Document Q&A", page_icon=im, layout="wide")

# --- Initialize State ---
if 'final_report' not in st.session_state:
    st.session_state.final_report = None

# --- Main App ---

# 1. Image
# Ensure you have an image at 'qna_rag/assets/dashboard_header.png'
# For now, we'll use a placeholder URL, but you should replace it.
st.image("src/torch_rag/AI8.jpg", width=1200)

# 2. Descriptive Text
st.title("Document Q&A Agents")
st.markdown("""
    This application leverages a multi-agent **CrewAI** to perform comprehensive research of your question or query on a defined set of documents.
""")

# 3. User Input (Text Area)
user_question = st.text_area(
    "Input your question or query:",
    height=150,
    help="Enter a question or query you want researched from the data."
)

# 4. Submit Button
col1, col2 = st.columns([1, 4])
with col1:
    submit_button = st.button("Kickoff Analysis")

# Divider
st.divider()


# --- Integration with CrewAI ---

if submit_button:
    st.session_state.final_report = None # Clear previous report
    
    inputs = {
        'question': user_question
    }
    
    with st.status("Agents are researching...", expanded=True) as status:
        try:
            # Kickoff the crew
            # We capture the string returned by kickoff
            from torch_rag.tools.output_handler import capture_output
            with capture_output(status):
                final_result = run_from_gui(inputs)
            
            # Store in session state for the download button to see
            st.session_state.final_report = str(final_result)
            
            status.update(label="Analysis Complete!", state="complete", expanded=False)
        except Exception as e:
            status.update(label="Error occurred", state="error")
            st.error(f"Error: {e}")

# 6. Download Button
st.divider()

if st.session_state.final_report:
    st.success("Document Q&A report is ready for download.")
    
    # 1. Show a preview of the report in the app
    with st.expander("Preview Final Report"):
        st.markdown(st.session_state.final_report)
    
    # 2. Provide the download button using the data in memory
    st.download_button(
        label="Download Report",
        data=st.session_state.final_report,
        file_name="QnA_Report.json",
        mime="application/json",
        key="download_report_btn" # Unique key to prevent reset loops
    )
else:
    st.info("The download button will appear here once the agents finish their analysis.")