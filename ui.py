import streamlit as st
from app import load_and_split_document, create_vectorstore, create_qa_chain, ask_question
import os

st.set_page_config(page_title="Security Docs Q&A Bot", page_icon="🔒")
st.title("Security Docs Q&A Bot")
st.write("Upload your security document and ask questions about it.")

# Session state — so the vectorstore isn't rebuilt on every page reload
if "qa_ready" not in st.session_state:
    st.session_state.qa_ready = False

# Step 1: File upload
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file is not None:
    if st.button("Process Document"):
        with st.spinner("Processing document... (this may take a moment)"):
            # Temporarily save uploaded file
            os.makedirs("data", exist_ok=True)
            temp_path = os.path.join("data", uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Run the pipeline
            chunks = load_and_split_document(temp_path)
            vectorstore = create_vectorstore(chunks)
            retriever, llm = create_qa_chain(vectorstore)

            # Save in session state
            st.session_state.retriever = retriever
            st.session_state.llm = llm
            st.session_state.qa_ready = True

        st.success(f"Document ready! ({len(chunks)} chunks processed)")

# Step 2: Question asking
if st.session_state.qa_ready:
    query = st.text_input("Ask your question:")

    if query:
        with st.spinner("Searching for the answer..."):
            answer, sources = ask_question(
                st.session_state.retriever,
                st.session_state.llm,
                query
            )
        st.write("### Answer:")
        st.write(answer)

        with st.expander("View source chunks"):
            for i, doc in enumerate(sources):
                st.write(f"**Chunk {i+1}:**")
                st.write(doc.page_content)
                st.divider()