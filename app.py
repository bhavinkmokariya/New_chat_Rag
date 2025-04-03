import streamlit as st
import boto3
import faiss
import tempfile

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


# Function to download FAISS index file from S3 and count documents
def count_documents_in_faiss(bucket_name, index_path, file_name):
    """
    Downloads a FAISS index file from S3 and counts the number of documents.

    Args:
        bucket_name (str): The name of the S3 bucket.
        index_path (str): The path to the FAISS index in the S3 bucket.
        file_name (str): The name of the FAISS index file.

    Returns:
        int: The number of documents in the FAISS index.
    """
    s3 = create_s3_client()
    try:
        # Download FAISS file to a temporary location
        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            s3.download_file(bucket_name, f"{index_path}{file_name}", temp_file.name)
            # Load FAISS index
            index = faiss.read_index(temp_file.name)
            # Return document count
            return index.ntotal
    except Exception as e:
        st.error(f"Error loading FAISS index file: {e}")
        return None


# Streamlit UI
st.title("FAISS Index Document Counter")

with st.sidebar:
    st.header("Settings")
    proforma_index_path = st.text_input("Proforma Index Path", value="faiss_indexes/proforma_faiss_index/")
    po_index_path = st.text_input("PO Index Path", value="faiss_indexes/po_faiss_index/")
    faiss_file_name = st.text_input("FAISS File Name", value="index.faiss")

if st.button("Count Documents"):
    # Count documents for Proforma FAISS index
    proforma_document_count = count_documents_in_faiss(bucket_name, proforma_index_path, faiss_file_name)

    # Count documents for PO FAISS index
    po_document_count = count_documents_in_faiss(bucket_name, po_index_path, faiss_file_name)

    # Display results
    st.header("Document Counts")

    if proforma_document_count is not None:
        st.write(f"Proforma FAISS Index Document Count: {proforma_document_count}")
    else:
        st.write("Proforma FAISS Index Document Count: Error occurred")

    if po_document_count is not None:
        st.write(f"PO FAISS Index Document Count: {po_document_count}")
    else:
        st.write("PO FAISS Index Document Count: Error occurred")
