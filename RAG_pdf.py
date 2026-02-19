import streamlit as st
import tempfile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings

st.title("PDF RAG Chat")
load_dotenv()
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    print(len(chunks))
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity" ,search_kwargs={"k": 4})

    st.success("PDF Processed Successfully ✅")

    question = st.text_input("Ask a question about the PDF")

    if question:

        docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = PromptTemplate(
            template="""
            You are a helpful assistant.
            Answer ONLY from the provided transcript context.
            If the context is insufficient, just say you don't know.

            {context}
            Question: {question}
            """,
            input_variables = ['context', 'question']
        )
        llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen3-Coder-Next-FP8",
        task="text-generation",
        max_new_tokens=300)

        final_prompt = prompt.invoke({"context":context, "question":question})
        model = ChatHuggingFace(llm=llm)
    if st.button("Send"):
        result=model.invoke(final_prompt)
        st.write(result.content)

