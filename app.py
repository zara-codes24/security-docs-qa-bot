from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
import streamlit as st
from rag_pipeline import get_qa_chain

st.set_page_config(page_title="Security Docs Q&A Bot")
st.title("Security Docs Q&A Bot")

# Store the QA chain in session state so it doesn't reload on every interaction
if "qa_chain" not in st.session_state:
    with st.spinner("Loading documents into vector database..."):
        st.session_state.qa_chain = get_qa_chain()

question = st.text_input("Ask a question about your documents:")

if question:
    with st.spinner("Analyzing sources and generating answer..."):
        result = st.session_state.qa_chain.invoke({"query": question})
        st.write("### Answer:")
        st.write(result["result"])

        with st.expander("View Source Documents"):
            for doc in result["source_documents"]:
                st.write(doc.page_content[:300] + "...")