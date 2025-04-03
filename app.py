import streamlit as st
import boto3
import faiss
import tempfile
import toml
import os
from PO_s3store import PO_INDEX_PATH, S3_BUCKET as PO_S3_BUCKET
from proforma_s3store import S3_PROFORMA_INDEX_PATH, S3_BUCKET as PROFORMA_S3_BUCKET

# Configuration constants from the provided scripts
#SECRETS_FILE_PATH = "C:/Users/Admin/.vscode/s3/.streamlit/secrets.toml"

# Load secrets from secrets.toml
try:
    #secrets = toml.load(SECRETS_FILE_PATH)
    AWS_ACCESS_KEY = st.secrets["access_key_id"]
    AWS_SECRET_KEY = st.secrets["secret_access_key"]
except FileNotFoundError:
    st.error("Secrets file not found. Please ensure secrets.toml is correctly configured.")
    st.stop()


# Function to create an S3 client using the provided credentials
def create_s3_client():
    """Creates an authenticated S3 client."""
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
    )


# Function to download FAISS index file from S3 and count documents
def count_documents_in_faiss(bucket_name, index_path, file_name="index.faiss"):
    """
    Downloads a FAISS index file from S3 and counts the number of documents.

    Args:
        bucket_name (str): The name of the S3 bucket.
        index_path (str): The path to the FAISS index in the S3 bucket.
        file_name (str): The name of the FAISS index file (default: 'index.faiss').

    Returns:
        int: The number of documents in the FAISS index, or None if an error occurs.
    """
    s3 = create_s3_client()
    full_s3_path = f"{index_path}{file_name}"
    try:
        # Check if the FAISS index file exists in S3
        s3.head_object(Bucket=bucket_name, Key=full_s3_path)

        # Download FAISS file to a temporary location
        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            s3.download_file(bucket_name, full_s3_path, temp_file.name)
            # Load FAISS index
            index = faiss.read_index(temp_file.name)
            # Return document count
            return index.ntotal
    except s3.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            st.warning(f"FAISS index file not found at {full_s3_path}")
        else:
            st.error(f"Error accessing S3: {e}")
        return None
    except Exception as e:
        st.error(f"Error loading FAISS index file: {e}")
        return None


# Streamlit UI
st.title("FAISS Index Document Counter")

# Sidebar for configuration
with st.sidebar:
    st.header("Settings")
    proforma_index_path = st.text_input("Proforma Index Path", value=S3_PROFORMA_INDEX_PATH)
    po_index_path = st.text_input("PO Index Path", value=PO_INDEX_PATH)
    faiss_file_name = st.text_input("FAISS File Name", value="index.faiss")
    bucket_name = st.text_input("S3 Bucket Name", value=PO_S3_BUCKET)

# Button to trigger document counting
if st.button("Count Documents"):
    # Count documents for Proforma FAISS index
    proforma_document_count = count_documents_in_faiss(bucket_name, proforma_index_path, faiss_file_name)

    # Count documents for PO FAISS index
    po_document_count = count_documents_in_faiss(bucket_name, po_index_path, faiss_file_name)

    # Display results
    st.header("Document Counts")

    if proforma_document_count is not None:
        st.success(f"Proforma FAISS Index Document Count: {proforma_document_count}")
    else:
        st.warning("Proforma FAISS Index Document Count: Not available (check logs or S3 path)")

    if po_document_count is not None:
        st.success(f"PO FAISS Index Document Count: {po_document_count}")
    else:
        st.warning("PO FAISS Index Document Count: Not available (check logs or S3 path)")

# Optional: Display last updated time (assuming scheduler runs daily at midnight)
st.write(f"Last checked: {st.session_state.get('last_checked', 'Not yet checked')}")
if 'last_checked' not in st.session_state:
    st.session_state['last_checked'] = "Not yet checked"

# Note about scheduler
st.info("Note: The FAISS indices are updated daily at 12:00 AM and checked every 10 minutes by the scheduler.")

# Instructions for running the app
st.markdown("""
### How to Use
1. Ensure your `secrets.toml` file is correctly set up with AWS credentials.
2. Run the scheduler script in the background to keep FAISS indices updated.
3. Click 'Count Documents' to see the number of indexed documents in the Proforma and PO FAISS indices.
""")