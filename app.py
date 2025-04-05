import boto3
import os
import logging
import tempfile
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import toml

# Configuration constants
S3_BUCKET = "kalika-rag"
PO_INDEX_PATH = "faiss_indexes/po_faiss_index/"  # Path for PO FAISS index
PROFORMA_INDEX_PATH = "faiss_indexes/proforma_faiss_index/"  # Path for Proforma FAISS index
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Load secrets from Streamlit Cloud (assumes secrets are available in st.secrets)
try:
    AWS_ACCESS_KEY = st.secrets["access_key_id"]
    AWS_SECRET_KEY = st.secrets["secret_access_key"]
except KeyError as e:
    logging.error(f"Missing secret: {e}")
    st.error(f"Missing secret: {e}. Please ensure credentials are set in Streamlit Cloud secrets.")
    st.stop()

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


def load_faiss_index_from_s3(bucket, prefix):
    """Load FAISS index from S3."""
    try:
        # List objects in the specified S3 prefix
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)

        if 'Contents' not in response:
            logging.warning(f"No FAISS index files found in S3 at {prefix}")
            return None

        # Use a temporary directory to download and load the index
        with tempfile.TemporaryDirectory() as temp_dir:
            index_files = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith(('.faiss', '.pkl'))]
            if not index_files:
                logging.warning(f"No valid FAISS index files (.faiss or .pkl) found in {prefix}")
                return None

            # Download all index-related files
            for s3_key in index_files:
                local_path = os.path.join(temp_dir, os.path.basename(s3_key))
                s3_client.download_file(bucket, s3_key, local_path)
                logging.info(f"Downloaded {s3_key} to {local_path}")

            # Load the FAISS index
            index_name = os.path.splitext(os.path.basename(index_files[0]))[0]  # Assume first file name as base
            vector_store = FAISS.load_local(temp_dir, index_name, embeddings, allow_dangerous_deserialization=True)
            logging.info(f"Successfully loaded FAISS index from S3 at {prefix}")
            return vector_store

    except Exception as e:
        logging.error(f"Failed to load FAISS index from S3 at {prefix}: {str(e)}")
        return None


def main():
    st.title("FAISS Index Chatbot")

    # Select which index to load
    index_type = st.selectbox("Select FAISS Index to Load", ["PO Index", "Proforma Index"])
    index_path = PO_INDEX_PATH if index_type == "PO Index" else PROFORMA_INDEX_PATH

    if st.button("Load FAISS Index"):
        with st.spinner("Loading FAISS index from S3..."):
            vector_store = load_faiss_index_from_s3(S3_BUCKET, index_path)

        if vector_store:
            st.success(f"FAISS index successfully loaded from S3 at {index_path}!")
            # Optionally store the vector_store in session state for further use
            st.session_state['vector_store'] = vector_store
        else:
            st.error(f"Failed to load FAISS index from S3 at {index_path}. Check logs for details.")

    # Placeholder for future chatbot interaction
    if 'vector_store' in st.session_state:
        st.write("FAISS index is loaded and ready for querying!")
        # Add your chatbot query logic here in future iterations


if __name__ == "__main__":
    main()