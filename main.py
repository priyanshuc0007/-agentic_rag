import os
import gradio as gr
import PyPDF2
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY")

# Initialize LLM
llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.5)
# Set up embedding model
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + " "
    return text

def setup_vector_db(pdf_texts):
    docs = pdf_texts.split(". ")  # Simple chunking
    vector_store = FAISS.from_texts(docs, embedding_model)
    return vector_store

# Memory for conversation history
memory = ConversationBufferMemory(memory_key="chat_history")

# PDF Retrieval Tool
def pdf_retriever_tool(query: str):
    retriever = vector_db.as_retriever()
    return retriever.get_relevant_documents(query)

retriever_tool = Tool(
    name="PDFRetriever",
    func=pdf_retriever_tool,
    description="Retrieves relevant content from uploaded PDFs based on user queries."
)

# Initialize Agent
agent = initialize_agent(
    tools=[retriever_tool],
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

def chat_agent(input_text):
    response = agent.run(input_text)
    return response

# Gradio UI
def gradio_chatbot(user_input):
    response = chat_agent(user_input)
    return response

interface = gr.Interface(fn=gradio_chatbot, inputs="text", outputs="text", title="RAG Agent Chatbot")

if __name__ == "__main__":
    # Load and process PDFs before running the chatbot
    pdf_text = extract_text_from_pdf(r"C:\Users\Hp\Desktop\pdf_chatbot\data\Hindu-Review-April-2024.pdf")
    vector_db = setup_vector_db(pdf_text)
    interface.launch()
