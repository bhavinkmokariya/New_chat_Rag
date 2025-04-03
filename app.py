import streamlit as st
import boto3
import faiss
import tempfile
import toml
import os
import numpy as np
from PO_s3store import PO_INDEX_PATH, S3_BUCKET as PO_S3_BUCKET
from proforma_s3store import S3_PROFORMA_INDEX_PATH, S3_BUCKET as PROFORMA_S3_BUCKET
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline

# Configuration constants
#SECRETS_FILE_PATH = "C:/Users/Admin/.vscode/s3/.streamlit/secrets.toml"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Same as in your scripts

# Load secrets from secrets.toml
try:
    #secrets = toml.load(SECRETS_FILE_PATH)
    AWS_ACCESS_KEY = st.secrets["access_key_id"]
    AWS_SECRET_KEY = st.secrets["secret_access_key"]
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

# Function to download and load FAISS index directly from S3
def load_raw_faiss_index(bucket_name, index_path, file_name="index.faiss"):
    """
    Downloads and loads a raw FAISS index from S3.

    Args:
        bucket_name (str): The name of the S3 bucket.
        index_path (str): The path to the FAISS index in the S3 bucket.
        file_name (str): The name of the FAISS index file.

    Returns:
        faiss.Index: Loaded FAISS index, or None if an error occurs.
    """
    s3 = create_s3_client()
    full_s3_path = f"{index_path}{file_name}"
    try:
        s3.head_object(Bucket=bucket_name, Key=full_s3_path)
        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            s3.download_file(bucket_name, full_s3_path, temp_file.name)
            index = faiss.read_index(temp_file.name)
            return index
    except Exception as e:
        st.error(f"Error loading FAISS index from {full_s3_path}: {e}")
        return None

# Function to count documents in FAISS index
def count_documents_in_faiss(bucket_name, index_path, file_name="index.faiss"):
    """Counts the number of documents in a FAISS index."""
    index = load_raw_faiss_index(bucket_name, index_path, file_name)
    if index:
        return index.ntotal
    return None

# Function to load embeddings model
def get_embeddings_model():
    """Loads the embeddings model."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )

# RAG setup
def setup_rag_pipeline():
    """Sets up a text generation pipeline."""
    try:
        generator = pipeline("text-generation", model="distilgpt2", max_length=150)
        return generator
    except Exception as e:
        st.error(f"Error setting up RAG pipeline: {e}")
        return None

# Function to query FAISS index and generate response
def query_rag_raw_faiss(index, question, embeddings_model, generator, texts, k=3):
    """
    Queries the raw FAISS index and generates a response.

    Args:
        index: Loaded FAISS index.
        question (str): User's question.
        embeddings_model: Embeddings model for query encoding.
        generator: Text generation pipeline.
        texts (list): List of original text chunks (for retrieval).
        k (int): Number of documents to retrieve.

    Returns:
        str: Generated response.
    """
    if not index or not embeddings_model or not generator or not texts:
        return "Error: RAG components not properly initialized."

    try:
        # Embed the question
        question_embedding = embeddings_model.embed_query(question)
        question_embedding = np.array([question_embedding]).astype('float32')

        # Search the FAISS index
        distances, indices = index.search(question_embedding, k)
        if len(indices) == 0 or max(indices[0]) >= len(texts):
            return "No relevant documents found."

        # Retrieve relevant text chunks
        context = "\n".join([texts[idx] for idx in indices[0] if idx < len(texts)])

        # Generate response
        prompt = f"Based on the following context:\n{context}\n\nAnswer the question: {question}"
        response = generator(prompt, max_length=150, num_return_sequences=1)[0]['generated_text']
        return response
    except Exception as e:
        return f"Error generating response: {e}"

# Load text chunks from S3 (simplified assumption: stored alongside index)
def load_text_chunks(bucket_name, folder_path):
    """
    Loads text chunks from S3 (assumes a text file with chunks exists).

    Args:
        bucket_name (str): S3 bucket name.
        folder_path (str): Folder path in S3.

    Returns:
        list: List of text chunks.
    """
    s3 = create_s3_client()
    try:
        # Assuming a file like 'chunks.txt' exists with one chunk per line
        response = s3.get_object(Bucket=bucket_name, Key=f"{folder_path}chunks.txt")
        chunks = response['Body'].read().decode('utf-8').splitlines()
        return chunks
    except Exception as e:
        st.warning(f"Could not load text chunks: {e}. Using placeholder.")
        return ["Placeholder text"]  # Fallback if no chunks file exists

# Streamlit UI
st.title("FAISS Index Document Counter & RAG Query Tool")

# Sidebar for configuration
with st.sidebar:
    st.header("Settings")
    proforma_index_path = st.text_input("Proforma Index Path", value=S3_PROFORMA_INDEX_PATH)
    po_index_path = st.text_input("PO Index Path", value=PO_INDEX_PATH)
    faiss_file_name = st.text_input("FAISS File Name", value="index.faiss")
    bucket_name = st.text_input("S3 Bucket Name", value=PO_S3_BUCKET)

# Tabbed interface
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
            folder_path = "proforma_invoice/" if index_type == "Proforma" else "PO_Dump/"
            faiss_index = load_raw_faiss_index(bucket_name, index_path, faiss_file_name)

            if faiss_index:
                # Load embeddings and generator
                embeddings_model = get_embeddings_model()
                generator = setup_rag_pipeline()

                # Load text chunks (assumes a companion file exists)
                texts = load_text_chunks(bucket_name, folder_path)

                if embeddings_model and generator:
                    response = query_rag_raw_faiss(faiss_index, question, embeddings_model, generator, texts)
                    st.write("**Answer:**")
                    st.write(response)
                else:
                    st.error("Failed to initialize RAG components.")
            else:
                st.error(f"Failed to load {index_type} FAISS index.")
        else:
            st.warning("Please enter a question.")

# Footer
st.info("Note: The FAISS indices are updated daily at 12:00 AM and checked every 10 minutes by the scheduler.")
st.markdown("""
### How to Use
- **Document Count**: Click 'Count Documents' to see the number of indexed documents.
- **RAG Query**: Enter a question, select an index, and click 'Get Answer' to query the documents.
""")