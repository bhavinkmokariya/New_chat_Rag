import streamlit as st
import boto3
import os
import tempfile
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Configuration constants
S3_BUCKET = st.secrets["S3_BUCKET_NAME"]
PO_INDEX_PATH = "faiss_indexes/po_faiss_index/"
PROFORMA_INDEX_PATH = "faiss_indexes/proforma_faiss_index/"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GOOGLE_MODEL = "gemini-1.5-pro"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Load secrets from Streamlit Cloud
AWS_ACCESS_KEY = st.secrets["access_key_id"]
AWS_SECRET_KEY = st.secrets["secret_access_key"]
GOOGLE_API_KEY = st.secrets['gemini_api_key']  # Ensure this matches your Streamlit Cloud secret

# Initialize S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
)

# Initialize embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

# Initialize Google Gemini 1.5 Pro
llm = ChatGoogleGenerativeAI(
    model=GOOGLE_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7
)

def load_faiss_index_from_s3(bucket, prefix):
    """Load the latest FAISS index from S3."""
    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        if 'Contents' not in response:
            logging.warning(f"No FAISS index found at {prefix}")
            return None

        faiss_files = [obj for obj in response['Contents'] if obj['Key'].endswith('.faiss')]
        if not faiss_files:
            logging.warning(f"No .faiss files found at {prefix}")
            return None

        latest_file = max(faiss_files, key=lambda x: x['LastModified'])
        faiss_key = latest_file['Key']

        with tempfile.TemporaryDirectory() as temp_dir:
            faiss_local_path = os.path.join(temp_dir, "faiss_index.faiss")
            pkl_local_path = os.path.join(temp_dir, "faiss_index.pkl")

            s3_client.download_file(bucket, faiss_key, faiss_local_path)
            pkl_key = faiss_key.replace(".faiss", ".pkl")
            s3_client.download_file(bucket, pkl_key, pkl_local_path)

            vector_store = FAISS.load_local(temp_dir, "faiss_index", embeddings, allow_dangerous_deserialization=True)
            logging.info(f"Loaded FAISS index from {faiss_key}")
            return vector_store

    except Exception as e:
        logging.error(f"Failed to load FAISS index from {prefix}: {str(e)}")
        return None

def merge_vector_stores(po_store, proforma_store):
    """Merge PO and Proforma FAISS vector stores into a single store."""
    if po_store and proforma_store:
        combined_store = FAISS.from_texts(
            po_store.get_texts() + proforma_store.get_texts(),
            embeddings
        )
        logging.info("Merged PO and Proforma FAISS indexes")
        return combined_store
    elif po_store:
        return po_store
    elif proforma_store:
        return proforma_store
    else:
        return None

# Load and merge FAISS indexes
po_vector_store = load_faiss_index_from_s3(S3_BUCKET, PO_INDEX_PATH)
proforma_vector_store = load_faiss_index_from_s3(S3_BUCKET, PROFORMA_INDEX_PATH)
vector_store = merge_vector_stores(po_vector_store, proforma_vector_store)

# Define the prompt template
prompt_template = """
You are a helpful assistant answering questions based on Purchase Order (PO) and Proforma Invoice documents.
Use the following context to provide accurate and concise answers. If the context doesn't contain relevant information, say so.

Context: {context}

Question: {question}

Answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Set up RetrievalQA chain
if vector_store:
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
else:
    qa_chain = None
    st.error("No FAISS index loaded. Chatbot functionality is limited.")

# Streamlit UI
st.title("PO & Proforma Invoice Chatbot")
st.write("Ask questions about Purchase Orders or Proforma Invoices!")

user_input = st.text_input("Your question:", "")

if user_input:
    if qa_chain:
        try:
            result = qa_chain({"query": user_input})
            answer = result["result"]
            source_docs = result["source_documents"]

            st.write("**Answer:**")
            st.write(answer)

            with st.expander("Source Documents"):
                for i, doc in enumerate(source_docs):
                    st.write(f"**Document {i + 1}:**")
                    st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)

        except Exception as e:
            st.error(f"Error processing your question: {str(e)}")
    else:
        st.write("Sorry, I can't answer questions without a loaded FAISS index.")

st.write("Powered by Gemini 1.5 Pro and FAISS embeddings.")