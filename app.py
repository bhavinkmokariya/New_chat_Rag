import streamlit as st
import boto3
import faiss
import tempfile
import toml
import os
from PO_s3store import PO_INDEX_PATH, S3_BUCKET as PO_S3_BUCKET
from proforma_s3store import S3_PROFORMA_INDEX_PATH, S3_BUCKET as PROFORMA_S3_BUCKET
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

# Configuration constants from the provided scripts
SECRETS_FILE_PATH = "C:/Users/Admin/.vscode/s3/.streamlit/secrets.toml"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Same as in your scripts

# Load secrets from secrets.toml
try:
    secrets = toml.load(SECRETS_FILE_PATH)
    AWS_ACCESS_KEY = secrets["access_key_id"]
    AWS_SECRET_KEY = secrets["secret_access_key"]
except FileNotFoundError:
    st.error("Secrets file not found. Please ensure secrets.toml is correctly configured.")
    st.stop()


# Function to create an S3 client
def create_s3_client():
    """Creates an authenticated S3 client."""
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
    )


# Function to download FAISS index file from S3 and load it
def load_faiss_index(bucket_name, index_path, file_name="index.faiss"):
    """
    Downloads and loads a FAISS index from S3.

    Args:
        bucket_name (str): The name of the S3 bucket.
        index_path (str): The path to the FAISS index in the S3 bucket.
        file_name (str): The name of the FAISS index file.

    Returns:
        LangchainFAISS: Loaded FAISS index, or None if an error occurs.
    """
    s3 = create_s3_client()
    full_s3_path = f"{index_path}{file_name}"
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
    try:
        s3.head_object(Bucket=bucket_name, Key=full_s3_path)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_index_path = os.path.join(temp_dir, "faiss_index")
            os.makedirs(temp_index_path, exist_ok=True)
            s3.download_file(bucket_name, full_s3_path, os.path.join(temp_index_path, file_name))
            index = LangchainFAISS.load_local(temp_index_path, embeddings, allow_dangerous_deserialization=True)
            return index
    except Exception as e:
        st.error(f"Error loading FAISS index from {full_s3_path}: {e}")
        return None


# Function to count documents in FAISS index
def count_documents_in_faiss(bucket_name, index_path, file_name="index.faiss"):
    """Counts the number of documents in a FAISS index."""
    index = load_faiss_index(bucket_name, index_path, file_name)
    if index:
        return index.index.ntotal
    return None


# RAG setup
def setup_rag_pipeline():
    """Sets up a RAG pipeline with a lightweight Hugging Face model."""
    try:
        # Use a lightweight model for text generation
        generator = pipeline("text-generation", model="distilgpt2", max_length=100)
        llm = HuggingFacePipeline(pipeline=generator)

        # Define a simple prompt template
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="Based on the following context:\n{context}\n\nAnswer the question: {question}"
        )
        return llm, prompt_template
    except Exception as e:
        st.error(f"Error setting up RAG pipeline: {e}")
        return None, None


# Function to query the FAISS index and generate a response
def query_rag(index, question, llm, prompt_template, k=3):
    """
    Queries the FAISS index and generates a response using RAG.

    Args:
        index: Loaded FAISS index.
        question (str): User's question.
        llm: Language model pipeline.
        prompt_template: Prompt template for formatting.
        k (int): Number of documents to retrieve.

    Returns:
        str: Generated response.
    """
    if not index or not llm or not prompt_template:
        return "Error: RAG components not properly initialized."

    try:
        # Retrieve relevant documents
        docs = index.similarity_search(question, k=k)
        context = "\n".join([doc.page_content for doc in docs])

        # Format prompt and generate response
        prompt = prompt_template.format(context=context, question=question)
        response = llm(prompt)
        return response
    except Exception as e:
        return f"Error generating response: {e}"


# Streamlit UI
st.title("FAISS Index Document Counter & RAG Query Tool")

# Sidebar for configuration
with st.sidebar:
    st.header("Settings")
    proforma_index_path = st.text_input("Proforma Index Path", value=S3_PROFORMA_INDEX_PATH)
    po_index_path = st.text_input("PO Index Path", value=PO_INDEX_PATH)
    faiss_file_name = st.text_input("FAISS File Name", value="index.faiss")
    bucket_name = st.text_input("S3 Bucket Name", value=PO_S3_BUCKET)

# Tabbed interface for Document Count and RAG Query
tab1, tab2 = st.tabs(["Document Count", "RAG Query"])

with tab1:
    st.header("Document Counts")
    if st.button("Count Documents"):
        proforma_document_count = count_documents_in_faiss(bucket_name, proforma_index_path, faiss_file_name)
        po_document_count = count_documents_in_faiss(bucket_name, po_index_path, faiss_file_name)

        if proforma_document_count is not None:
            st.success(f"Proforma FAISS Index Document Count: {proforma_document_count}")
        else:
            st.warning("Proforma FAISS Index Document Count: Not available")

        if po_document_count is not None:
            st.success(f"PO FAISS Index Document Count: {po_document_count}")
        else:
            st.warning("PO FAISS Index Document Count: Not available")

with tab2:
    st.header("Ask a Question")
    question = st.text_input("Enter your question about Proforma or PO documents:")
    index_type = st.selectbox("Select Index to Query", ["Proforma", "PO"])

    if st.button("Get Answer"):
        if question:
            # Load the appropriate FAISS index
            index_path = proforma_index_path if index_type == "Proforma" else po_index_path
            faiss_index = load_faiss_index(bucket_name, index_path, faiss_file_name)

            if faiss_index:
                # Setup RAG pipeline
                llm, prompt_template = setup_rag_pipeline()
                if llm and prompt_template:
                    response = query_rag(faiss_index, question, llm, prompt_template)
                    st.write("**Answer:**")
                    st.write(response)
                else:
                    st.error("Failed to initialize RAG pipeline.")
            else:
                st.error(f"Failed to load {index_type} FAISS index.")
        else:
            st.warning("Please enter a question.")

# Footer
st.info("Note: The FAISS indices are updated daily at 12:00 AM and checked every 10 minutes by the scheduler.")
st.markdown("""
### How to Use
- **Document Count**: Click 'Count Documents' to see the number of indexed documents.
- **RAG Query**: Enter a question, select an index (Proforma or PO), and click 'Get Answer' to query the documents.
""")