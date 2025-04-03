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


# Function to count FAISS index files and documents in S3
def count_faiss_files_and_documents(bucket_name, index_path):
    """
    Counts the number of FAISS index files and documents in an S3 bucket under a specific path.

    Args:
        bucket_name (str): The name of the S3 bucket.
        index_path (str): The path to the FAISS index in the S3 bucket.

    Returns:
        tuple: (int, int) - The number of FAISS index files and document count.
    """
    s3 = create_s3_client()
    try:
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket_name, Prefix=index_path)

        file_count = 0
        document_count = 0

        for page in pages:
            if "Contents" in page:
                for obj in page["Contents"]:
                    file_count += 1
                    if obj["Key"].endswith(".faiss"):
                        # Extract document count from FAISS file names (assuming format includes document count)
                        try:
                            doc_count = int(obj["Key"].split("_")[1].split(".")[0])
                            document_count += doc_count
                        except (IndexError, ValueError):
                            st.error(f"Error parsing document count from file: {obj['Key']}")

        return file_count, document_count
    except Exception as e:
        st.error(f"Error counting FAISS index files and documents: {e}")
        return None, None


# Streamlit UI
st.title("FAISS Index File and Document Counter")

with st.sidebar:
    st.header("Settings")
    proforma_index_path = st.text_input("Proforma Index Path", value="faiss_indexes/proforma_faiss_index/")
    po_index_path = st.text_input("PO Index Path", value="faiss_indexes/po_faiss_index/")

if st.button("Count Files and Documents"):
    # Count files and documents for Proforma FAISS index
    proforma_file_count, proforma_document_count = count_faiss_files_and_documents(bucket_name, proforma_index_path)

    # Count files and documents for PO FAISS index
    po_file_count, po_document_count = count_faiss_files_and_documents(bucket_name, po_index_path)

    # Display results
    st.header("Counts")

    if proforma_file_count is not None:
        st.write(f"Proforma FAISS Index File Count: {proforma_file_count}")
        st.write(f"Proforma Document Count: {proforma_document_count}")
    else:
        st.write("Proforma FAISS Index File Count: Error occurred")

    if po_file_count is not None:
        st.write(f"PO FAISS Index File Count: {po_file_count}")
        st.write(f"PO Document Count: {po_document_count}")
    else:
        st.write("PO FAISS Index File Count: Error occurred")
