import boto3
import os
import tempfile
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Configuration constants
S3_BUCKET = "kalika-rag"
PO_INDEX_PATH = "faiss_indexes/po_faiss_index/"  # Matches PO_s3store.py
PROFORMA_INDEX_PATH = "faiss_indexes/proforma_faiss_index/"  # Matches proforma_s3store.py
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)


def initialize_s3_client(aws_access_key, aws_secret_key):
    """Initialize and return an S3 client with provided credentials."""
    return boto3.client(
        "s3",
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
    )


# Initialize embeddings model (must match the one used to create the index)
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)


def load_faiss_index_from_s3(s3_client, bucket, index_prefix):
    """
    Load a FAISS index from S3 into memory and count the number of files loaded.

    Args:
        s3_client: Initialized boto3 S3 client
        bucket (str): S3 bucket name
        index_prefix (str): S3 prefix where FAISS index files are stored

    Returns:
        tuple: (FAISS vector store object or None, number of files loaded)
    """
    try:
        # Create a temporary directory to store downloaded index files
        with tempfile.TemporaryDirectory() as temp_dir:
            # List all files in the S3 prefix
            response = s3_client.list_objects_v2(Bucket=bucket, Prefix=index_prefix)
            if 'Contents' not in response:
                logging.warning(f"No FAISS index files found at {index_prefix}")
                return None, 0

            # Count and download all index files
            file_count = 0
            for obj in response['Contents']:
                key = obj['Key']
                file_name = os.path.basename(key)
                local_path = os.path.join(temp_dir, file_name)
                logging.info(f"Downloading {key} to {local_path}")
                s3_client.download_file(bucket, key, local_path)
                file_count += 1

            # Load the FAISS index from the local directory
            vector_store = FAISS.load_local(temp_dir, embeddings, allow_dangerous_deserialization=True)
            logging.info(f"Successfully loaded FAISS index from {index_prefix} with {file_count} files")
            return vector_store, file_count

    except Exception as e:
        logging.error(f"Failed to load FAISS index from {index_prefix}: {str(e)}")
        return None, 0


def load_all_faiss_indexes(s3_client):
    """
    Load all FAISS indexes from S3 for PO and proforma prefixes.

    Args:
        s3_client: Initialized boto3 S3 client

    Returns:
        dict: Mapping of index prefixes to (vector_store, file_count) tuples
    """
    index_prefixes = {
        "PO": PO_INDEX_PATH,
        "Proforma": PROFORMA_INDEX_PATH
    }

    loaded_indexes = {}
    for name, prefix in index_prefixes.items():
        vector_store, file_count = load_faiss_index_from_s3(s3_client, S3_BUCKET, prefix)
        loaded_indexes[name] = (vector_store, file_count)

    return loaded_indexes


# Streamlit app integration
import streamlit as st

st.title("FAISS Index Loader")

# Load AWS credentials from Streamlit secrets
try:
    aws_access_key = st.secrets["access_key_id"]
    aws_secret_key = st.secrets["secret_access_key"]
except KeyError as e:
    st.error(f"Missing secret: {e}. Please configure secrets in Streamlit Cloud.")
    st.stop()

# Initialize S3 client
s3_client = initialize_s3_client(aws_access_key, aws_secret_key)


# Cache the index loading
@st.cache_resource
def cached_load_all_faiss_indexes(_s3_client):
    return load_all_faiss_indexes(_s3_client)


# Load all FAISS indexes
loaded_indexes = cached_load_all_faiss_indexes(s3_client)

# Display results
for name, (vector_store, file_count) in loaded_indexes.items():
    if vector_store:
        st.success(f"{name} FAISS Index Loaded Successfully! ({file_count} files)")
    else:
        st.error(f"Failed to load {name} FAISS Index")

# Optional: Test similarity search
query = st.text_input("Enter a test query:")
if query:
    for name, (vector_store, file_count) in loaded_indexes.items():
        if vector_store:
            results = vector_store.similarity_search(query, k=3)
            st.write(f"Top {name} Results:", results)
        else:
            st.warning(f"No {name} index available to process the query.")