from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()


def load_and_split_document(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    return chunks


def create_vectorstore(chunks, persist_directory="chroma_db"):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print(f"Vectorstore created with {len(chunks)} chunks")
    return vectorstore


def create_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Please check your .env file.")

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=api_key,
        temperature=0
    )

    return retriever, llm


def ask_question(retriever, llm, query):
    if not query or not query.strip():
        return "Please enter a valid question.", []

    try:
        docs = retriever.invoke(query)

        if not docs:
            return "No relevant information found in the document for this question.", []

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""Answer the question based only on the following context. 
If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {query}
Answer:"""

        response = llm.invoke(prompt)
        return response.content, docs

    except Exception as e:
        return f"An error occurred while generating the answer: {str(e)}", []


if __name__ == "__main__":
    chunks = load_and_split_document("data/WEF_Global_Cybersecurity_Outlook_2025.pdf")
    vectorstore = create_vectorstore(chunks)
    retriever, llm = create_qa_chain(vectorstore)

    answer, sources = ask_question(retriever, llm, "What are the main cybersecurity risks mentioned?")
    print("\nAnswer:", answer)