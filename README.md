# 📄 PDF RAG Chatbot

A Retrieval-Augmented Generation (RAG) based chatbot that allows users to upload any PDF file and chat with it.<br>
Built using LangChain, FAISS, Hugging Face Embeddings, and Streamlit, this application extracts text from uploaded PDFs, creates embeddings, and enables semantic question answering.

## 🚀 Features
📂 Upload any PDF file.<br>
📖 Extracts text using PyPDFLoader.<br>
✂ Splits text into chunks using RecursiveCharacterTextSplitter.<br>
🧠 Generates embeddings using sentence-transformers/all-MiniLM-L6-v2.<br>
🔎 Stores embeddings in FAISS vector database (in-memory).<br>
🤖 Uses Hugging Face LLM (Qwen/Qwen3-Coder-Next-FP8).<br>
💬 Interactive Streamlit chat interface.<br>
⚡ Real-time PDF processing.<br>

## 🏗️ Tech Stack.<br>
Python<br>
LangChain<br>
FAISS<br>
Hugging Face Hub<br>
Sentence Transformers<br>
Streamlit<br>
dotenv<br>



# ⚙️ Setup Instructions<br>
## 1️⃣ Clone Repository<br>
git clone https://github.com/your-username/your-repo-name.git<br>
cd your-repo-name<br>

## 2️⃣ Create Virtual Environment<br>
python -m venv venv<br>


Activate:<br>
Windows:
venv\Scripts\activate
<br>
Mac/Linux:
source venv/bin/activate

## 3️⃣ Install Dependencies<br>
pip install langchain langchain-community langchain-core langchain-huggingface faiss-cpu sentence-transformers streamlit python-dotenv pypdf<br>


## 4️⃣ Add Hugging Face API Token<br>
Create a .env file:
HUGGINGFACEHUB_API_TOKEN=your_token_here
<br>
Get your token from:
https://huggingface.co/settings/tokens

## 💬 Run the Application<br>
streamlit run app.py


Open in browser:<br>
http://localhost:8501

## 🧠 How It Works<br>

User uploads a PDF file<br>
PDF text is extracted using PyPDFLoader<br>
Text is split into chunks<br>
Embeddings are generated<br>
FAISS stores vector representations<br>
User question is converted to embedding<br>
Most similar chunks are retrieved<br>
Context + Question sent to LLM<br>
LLM generates final answer<br>
This follows a RAG (Retrieval-Augmented Generation) architecture.<br>



