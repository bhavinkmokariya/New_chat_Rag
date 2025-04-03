import streamlit as st
import boto3
import os
import tempfile
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load AWS credentials and S3 details from Streamlit secrets
aws_access_key_id = st.secrets["AWS_ACCESS_KEY_ID"]
aws_secret_access_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
bucket_name = st.secrets["s3_bucket_name"]
region_name = st.secrets["s3_region"]

# Function to create an S3 client using the provided credentials
def create_s3_client():
    """Creates an authenticated S3 client."""
    return boto3.client(
        "s3",
        region_name=region_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

# Function to load FAISS index from S3
def load_faiss_index_from_s3(bucket_name, index_path, embeddings):
    """
    Loads a FAISS index from S3.

    Args:
        bucket_name (str): The name of the S3 bucket.
        index_path (str): The path to the FAISS index in the S3 bucket.
        embeddings: The embeddings to use for the FAISS index.

    Returns:
        FAISS: The loaded FAISS index, or None if an error occurred.
    """
    s3 = create_s3_client()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            local_index_path = os.path.join(temp_dir, "faiss_index")
            os.makedirs(local_index_path, exist_ok=True)

            # Download the index files from S3
            paginator = s3.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=bucket_name, Prefix=index_path)
            for page in pages:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        file_key = obj["Key"]
                        file_name = os.path.basename(file_key)
                        local_file_path = os.path.join(local_index_path, file_name)
                        s3.download_file(bucket_name, file_key, local_file_path)

            # Load the FAISS index from the downloaded files
            index = FAISS.load_local(local_index_path, embeddings,allow_dangerous_deserialization=True)
            return index
    except Exception as e:
        st.error(f"Error loading FAISS index from S3: {e}")
        return None

# Initialize session state for counts
if "proforma_count" not in st.session_state:
    st.session_state.proforma_count = None
if "po_count" not in st.session_state:
    st.session_state.po_count = None

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    proforma_index_path = st.text_input("Proforma Index Path", value="faiss_indexes/proforma_faiss_index/")
    po_index_path = st.text_input("PO Index Path", value="faiss_indexes/po_faiss_index/")
    embedding_model_name = st.text_input("Embedding Model Name", value="sentence-transformers/all-MiniLM-L6-v2")

# Main app
st.title("FAISS Index Loader")

# Load embeddings (move this outside the button click for efficiency)

embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Button to load indexes and display counts
if st.button("Load FAISS Indexes"):
    # Load Proforma FAISS index
    proforma_index = load_faiss_index_from_s3(bucket_name, proforma_index_path, embeddings)
    if proforma_index:
        st.session_state.proforma_count = proforma_index.index.ntotal
    else:
        st.session_state.proforma_count = None

    # Load PO FAISS index
    po_index = load_faiss_index_from_s3(bucket_name, po_index_path, embeddings)
    if po_index:
        st.session_state.po_count = po_index.index.ntotal
    else:
        st.session_state.po_count = None

# Display counts
st.header("Index Counts")
if st.session_state.proforma_count is not None:
    st.write(f"Proforma FAISS Index Count: {st.session_state.proforma_count}")
else:
    st.write("Proforma FAISS Index Count: Not loaded")

if st.session_state.po_count is not None:
    st.write(f"PO FAISS Index Count: {st.session_state.po_count}")
else:
    st.write("PO FAISS Index Count: Not loaded")
