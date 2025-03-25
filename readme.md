# RAG Agent Chatbot

## Overview
This project is a **Retrieval-Augmented Generation (RAG) chatbot** that uses an **LLM (Llama 3.1-8B-Instant)** and **FAISS vector database** to retrieve relevant information from PDFs and provide context-aware responses. The chatbot is powered by **LangChain, Groq, and Gradio** for efficient document retrieval and interaction.

## Features
- **Extracts text from PDFs** using `PyPDF2`.
- **Creates a FAISS vector database** for efficient search and retrieval.
- **Uses Llama 3.1-8B-Instant (Groq API) as the LLM**.
- **Maintains conversation history** using `ConversationBufferMemory`.
- **Retrieves relevant information** from PDFs using a custom retrieval tool.
- **Integrates with Gradio** for an interactive UI.

## Installation
### Prerequisites
Ensure you have **Python 3.10+** installed and set up a virtual environment.

### Install Dependencies
```bash
pip install -r requirements.txt
```
#### Required Libraries:
- `langchain`
- `gradio`
- `PyPDF2`
- `faiss-cpu`
- `sentence-transformers`
- `langchain-groq`
- `python-dotenv`

## Environment Variables
Set up your API keys in a `.env` file:
```bash
GROQ_API_KEY=your_groq_api_key
```

## How It Works
1. **PDF Text Extraction:** Extracts text from the uploaded PDF.
2. **Vector Store Setup:** Converts text into embeddings using `SentenceTransformerEmbeddings` and stores them in a FAISS vector database.
3. **Retrieval Mechanism:** Searches relevant information using a `pdf_retriever_tool`.
4. **Conversational Agent:** Uses `ChatGroq` as the LLM and `ConversationBufferMemory` for context retention.
5. **Gradio UI:** Provides a chatbot interface for user interaction.

## Running the Chatbot
To start the chatbot, run:
```bash
python main.py
```
The chatbot will process the specified PDF and launch a **Gradio UI** where you can interact with it.

## Project Structure
```
📂 pdf_chatbot
 ├── 📂 data                      # Folder for storing PDFs
 ├── 📂 experiments               # Folder for testing models
 ├── 📜 main.py                   # Main script
 ├── 📜 requirements.txt           # Dependencies
 ├── 📜 .env                       # API keys (not committed to GitHub)
 └── 📜 README.md                  # Documentation
```

## Future Improvements
- Support for multiple PDFs.
- Enhance retrieval with chunking techniques.
- Add support for different LLMs like Claude or Mistral.


