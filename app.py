import streamlit as st
import boto3
import os
import logging
import tempfile
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Configuration constants
S3_BUCKET = "kalika-rag"
S3_PROFORMA_INDEX_PATH = "faiss_indexes/proforma_faiss_index/"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Initialize embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)


# Initialize S3 client using Streamlit secrets
def init_s3_client():
    try:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=st.secrets["access_key_id"],
            aws_secret_access_key=st.secrets["secret_access_key"],
        )
        return s3_client
    except Exception as e:
        logging.error(f"Failed to initialize S3 client: {str(e)}")
        st.error("Failed to connect to S3. Please check your credentials.")
        return None


# Load FAISS index from S3
def load_faiss_index_from_s3(s3_client):
    try:
        # List objects in the S3 index path
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_PROFORMA_INDEX_PATH)

        if 'Contents' not in response:
            logging.warning("No FAISS index found in S3.")
            return None

        # Create temporary directory to store downloaded index files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download all FAISS index files
            for obj in response['Contents']:
                key = obj['Key']
                local_path = os.path.join(temp_dir, os.path.basename(key))
                s3_client.download_file(S3_BUCKET, key, local_path)
                logging.info(f"Downloaded FAISS file: {key}")

            # Load the FAISS index
            vector_store = FAISS.load_local(temp_dir, embeddings, allow_dangerous_deserialization=True)
            logging.info("Successfully loaded FAISS index from S3")
            return vector_store

    except Exception as e:
        logging.error(f"Failed to load FAISS index from S3: {str(e)}")
        return None


# Main chatbot interface
def main():
    st.title("FAISS Index Loader Chatbot")

    # Initialize S3 client
    s3_client = init_s3_client()

    if s3_client:
        # Load FAISS index
        with st.spinner("Loading FAISS index from S3..."):
            vector_store = load_faiss_index_from_s3(s3_client)

        # Display result
        if vector_store:
            st.success("FAISS index successfully loaded from S3!")
            st.write(f"FAISS index is loaded from: s3://{S3_BUCKET}/{S3_PROFORMA_INDEX_PATH}")
        else:
            st.error("No FAISS index found in S3 or failed to load the index.")


if __name__ == "__main__":
    main()