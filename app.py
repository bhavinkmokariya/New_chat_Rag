import boto3
import os
import tempfile
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st

# Configuration constants
S3_BUCKET = "kalika-rag"
PO_INDEX_PATH = "faiss_indexes/po_faiss_index/"
PROFORMA_INDEX_PATH = "faiss_indexes/proforma_faiss_index/"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)


def initialize_s3_client(aws_access_key, aws_secret_key):
    """Initialize and return an S3 client with provided credentials."""
    try:
        return boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
        )
    except Exception as e:
        logging.error(f"Failed to initialize S3 client: {str(e)}")
        raise


# Initialize embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)


def load_faiss_index_from_s3(s3_client, bucket, index_prefix):
    """Load a FAISS index from S3 and count files."""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            response = s3_client.list_objects_v2(Bucket=bucket, Prefix=index_prefix)
            if 'Contents' not in response:
                logging.warning(f"No FAISS index files found at {index_prefix}")
                return None, 0

            file_count = 0
            for obj in response['Contents']:
                key = obj['Key']
                file_name = os.path.basename(key)
                local_path = os.path.join(temp_dir, file_name)
                logging.info(f"Downloading {key} to {local_path}")
                s3_client.download_file(bucket, key, local_path)
                file_count += 1

            vector_store = FAISS.load_local(temp_dir, embeddings, allow_dangerous_deserialization=True)
            logging.info(f"Loaded FAISS index from {index_prefix} with {file_count} files")
            return vector_store, file_count
    except Exception as e:
        logging.error(f"Failed to load FAISS index from {index_prefix}: {str(e)}")
        return None, 0


def load_all_faiss_indexes(s3_client):
    """Load all FAISS indexes from S3 and return total file count."""
    index_prefixes = {
        "PO": PO_INDEX_PATH,
        "Proforma": PROFORMA_INDEX_PATH
    }

    loaded_indexes = {}
    total_file_count = 0

    for name, prefix in index_prefixes.items():
        vector_store, file_count = load_faiss_index_from_s3(s3_client, S3_BUCKET, prefix)
        loaded_indexes[name] = vector_store
        total_file_count += file_count

    return loaded_indexes, total_file_count


# Streamlit app
st.title("FAISS Index Loader for Sales Team")

# Load AWS credentials from Streamlit secrets
try:
    aws_access_key = st.secrets["access_key_id"]
    aws_secret_key = st.secrets["secret_access_key"]
except KeyError as e:
    st.error(f"Missing secret: {e}. Please configure secrets in Streamlit Cloud.")
    st.stop()

# Initialize S3 client
try:
    s3_client = initialize_s3_client(aws_access_key, aws_secret_key)
except Exception as e:
    st.error(f"Failed to initialize S3 client: {str(e)}")
    st.stop()


# Cache the index loading
@st.cache_resource
def cached_load_all_faiss_indexes(_s3_client):
    try:
        result = load_all_faiss_indexes(_s3_client)
        if result is None or not isinstance(result, tuple):
            logging.error("load_all_faiss_indexes returned invalid result")
            return {}, 0
        return result
    except Exception as e:
        logging.error(f"Error in cached_load_all_faiss_indexes: {str(e)}")
        return {}, 0


# Load all FAISS indexes and get total file count
loaded_indexes, total_file_count = cached_load_all_faiss_indexes(s3_client)

# Display loading status and total file count
st.subheader("FAISS Index Loading Status")
if not isinstance(loaded_indexes, dict):
    st.error("Failed to load FAISS indexes: Invalid data returned. Check logs for details.")
    loaded_indexes = {}  # Reset to empty dict to prevent AttributeError

for name, vector_store in loaded_indexes.items():
    if vector_store:
        st.success(f"{name} FAISS Index Loaded Successfully!")
    else:
        st.error(f"Failed to load {name} FAISS Index")

st.info(f"Total FAISS Index Files Loaded: {total_file_count}")

# Optional: Test similarity search
query = st.text_input("Enter a test query:")
if query:
    for name, vector_store in loaded_indexes.items():
        if vector_store:
            try:
                results = vector_store.similarity_search(query, k=3)
                st.write(f"Top {name} Results:", results)
            except Exception as e:
                st.error(f"Error searching {name} index: {str(e)}")
        else:
            st.warning(f"No {name} index available to process the query.")