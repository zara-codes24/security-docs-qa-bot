# 🔒 Security Docs Q&A Bot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about security documents (PDFs) using semantic search and LLM-powered responses.

## 🎯 What It Does

Upload any security-related PDF (policy document, threat report, compliance guide) and ask natural language questions about it. The bot retrieves the most relevant sections and generates accurate, context-grounded answers — instead of relying on the LLM's general knowledge alone.

## 🛠️ Tech Stack

- **LangChain** — document processing and orchestration
- **ChromaDB** — vector database for storing embeddings
- **HuggingFace Sentence Transformers** — free local embeddings (`all-MiniLM-L6-v2`)
- **Groq API** — fast, free LLM inference (`llama-3.1-8b-instant`)
- **Streamlit** — interactive web UI

## 🏗️ How It Works

1. PDF is loaded and split into overlapping text chunks
2. Each chunk is converted into a vector embedding
3. Embeddings are stored in a local ChromaDB vector store
4. On each question, the top 3 most relevant chunks are retrieved
5. Retrieved context + question are sent to Groq's LLM to generate a grounded answer

## 🚀 Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/security-docs-qa-bot.git
cd security-docs-qa-bot

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Groq API key
# Create a .env file and add:
# GROQ_API_KEY=your_key_here

# 5. Run the app
streamlit run ui.py
```

## 📊 Evaluation

The bot was tested against 5 fixed questions to verify retrieval and answer quality. See [`evaluation_results.md`](evaluation_results.md) for full results.

## 📸 Screenshot

![Demo](screenshots/demo.png)

## 🔮 Future Improvements

- Support for multiple document uploads
- Conversation history / follow-up questions
- Automated evaluation metrics (precision, recall)

## 📝 License

This project is for educational and portfolio purposes.