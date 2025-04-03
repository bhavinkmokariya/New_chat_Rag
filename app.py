import streamlit as st
import boto3

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


# Function to count FAISS index files in S3
def count_faiss_files(bucket_name, index_path):
    """
    Counts the number of FAISS index files in an S3 bucket under a specific path.

    Args:
        bucket_name (str): The name of the S3 bucket.
        index_path (str): The path to the FAISS index in the S3 bucket.

    Returns:
        int: The number of files found under the specified path.
    """
    s3 = create_s3_client()
    try:
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket_name, Prefix=index_path)

        file_count = 0
        for page in pages:
            if "Contents" in page:
                file_count += len(page["Contents"])

        return file_count
    except Exception as e:
        st.error(f"Error counting FAISS index files: {e}")
        return None


# Streamlit UI
st.title("FAISS Index File Counter")

with st.sidebar:
    st.header("Settings")
    proforma_index_path = st.text_input("Proforma Index Path", value="faiss_indexes/proforma_faiss_index/")
    po_index_path = st.text_input("PO Index Path", value="faiss_indexes/po_faiss_index/")

if st.button("Count Files"):
    # Count files for Proforma FAISS index
    proforma_file_count = count_faiss_files(bucket_name, proforma_index_path)

    # Count files for PO FAISS index
    po_file_count = count_faiss_files(bucket_name, po_index_path)

    # Display results
    st.header("File Counts")
    if proforma_file_count is not None:
        st.write(f"Proforma FAISS Index File Count: {proforma_file_count}")
    else:
        st.write("Proforma FAISS Index File Count: Error occurred")

    if po_file_count is not None:
        st.write(f"PO FAISS Index File Count: {po_file_count}")
    else:
        st.write("PO FAISS Index File Count: Error occurred")
