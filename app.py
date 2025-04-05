import streamlit as st
import boto3
import os
import logging
import tempfile
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai

# Configuration constants
S3_BUCKET = "kalika-rag"
S3_PROFORMA_INDEX_PATH = "faiss_indexes/proforma_faiss_index/"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-1.5-pro"

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


# Initialize Gemini API
def init_gemini():
    try:
        genai.configure(api_key=st.secrets["gemini_api_key"])
        model = genai.GenerativeModel(GEMINI_MODEL)
        return model
    except Exception as e:
        logging.error(f"Failed to initialize Gemini: {str(e)}")
        st.error("Failed to connect to Gemini API. Please check your API key.")
        return None


# Load FAISS index from S3 and calculate size
def load_faiss_index_from_s3(s3_client):
    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_PROFORMA_INDEX_PATH)

        if 'Contents' not in response:
            logging.warning("No FAISS index found in S3.")
            return None

        with tempfile.TemporaryDirectory() as temp_dir:
            total_size = 0
            for obj in response['Contents']:
                key = obj['Key']
                local_path = os.path.join(temp_dir, os.path.basename(key))
                s3_client.download_file(S3_BUCKET, key, local_path)
                file_size = os.path.getsize(local_path)  # Get size in bytes
                total_size += file_size
                logging.info(f"Downloaded FAISS file: {key} ({file_size / 1024:.2f} KB)")

            # Convert total size to human-readable format
            if total_size < 1024:
                size_str = f"{total_size} bytes"
            elif total_size < 1024 * 1024:
                size_str = f"{total_size / 1024:.2f} KB"
            else:
                size_str = f"{total_size / (1024 * 1024):.2f} MB"

            st.write(f"Total size of FAISS index being loaded: {size_str}")
            logging.info(f"Total FAISS index size: {size_str}")

            vector_store = FAISS.load_local(temp_dir, embeddings, allow_dangerous_deserialization=True)
            logging.info("Successfully loaded FAISS index from S3")
            return vector_store

    except Exception as e:
        logging.error(f"Failed to load FAISS index from S3: {str(e)}")
        return None


# Query the FAISS index with adjustable k value
def query_faiss_index(vector_store, query, k=3):
    try:
        results = vector_store.similarity_search(query, k=k)
        logging.info(f"Retrieved {len(results)} documents from FAISS index")
        return results
    except Exception as e:
        logging.error(f"Error querying FAISS index: {str(e)}")
        return None


# Generate response using Gemini
def generate_response(model, query, faiss_results):
    try:
        if not faiss_results:
            return "No relevant information found in the proforma invoices."

        # Combine FAISS results into a context
        context = "\n\n".join([result.page_content for result in faiss_results])

        # Log the number of tokens/characters being sent to Gemini
        context_length = len(context)
        logging.info(f"Context size: ~{context_length} characters from {len(faiss_results)} documents")

        prompt = f"Based on the following information from proforma invoices:\n\n{context}\n\nAnswer the query: {query}"

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logging.error(f"Error generating response with Gemini: {str(e)}")
        return "An error occurred while generating the response."


# Main chatbot interface
def main():
    st.title("Proforma Invoice Chatbot with Gemini")

    # Initialize S3 client and Gemini model
    s3_client = init_s3_client()
    gemini_model = init_gemini()
    vector_store = None

    if s3_client:
        with st.spinner("Loading FAISS index from S3..."):
            vector_store = load_faiss_index_from_s3(s3_client)

        if vector_store:
            st.success("FAISS index successfully loaded from S3!")
            st.write(f"FAISS index is loaded from: s3://{S3_BUCKET}/{S3_PROFORMA_INDEX_PATH}")
        else:
            st.error("No FAISS index found in S3 or failed to load the index.")
            return

    if not gemini_model:
        return

    # Add sidebar for advanced settings
    with st.sidebar:
        st.header("Search Settings")

        # Add a slider for k value (number of documents to retrieve)
        k_value = st.slider(
            "Number of documents to search (k)",
            min_value=1,
            max_value=100,
            value=10,
            help="Higher values retrieve more documents but may include less relevant information"
        )

        st.info(
            "Recommended settings:\n"
            "- For specific questions: 3-5 documents\n"
            "- For broader questions: 10-20 documents\n"
            "- For comprehensive analysis: 50+ documents"
        )

    # Query input and response
    if vector_store:
        st.subheader("Ask a Question")
        query = st.text_input("Enter your query about the proforma invoices:")

        if st.button("Submit"):
            if query:
                with st.spinner(f"Searching across {k_value} documents and generating response..."):
                    # Search FAISS index with custom k value
                    faiss_results = query_faiss_index(vector_store, query, k=k_value)

                    # Generate response with Gemini
                    response = generate_response(gemini_model, query, faiss_results)

                st.subheader("Response")
                st.write(response)

                # Show information about the search
                st.subheader("Search Details")
                st.write(f"Retrieved {len(faiss_results) if faiss_results else 0} documents for this query")

                # Optionally show the documents that were used
                with st.expander("View documents used for this response"):
                    if faiss_results:
                        for i, doc in enumerate(faiss_results):
                            st.markdown(f"**Document {i + 1}**")
                            st.text(doc.page_content)
                            st.divider()
                    else:
                        st.write("No documents retrieved")
            else:
                st.warning("Please enter a query.")


if __name__ == "__main__":
    main()