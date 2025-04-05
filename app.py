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


# Initialize embeddings model with Hugging Face token
def init_embeddings():
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False},
            huggingfacehub_api_token=st.secrets["huggingface_token"]
        )
        return embeddings
    except Exception as e:
        logging.error(f"Failed to initialize embeddings: {str(e)}")
        st.error("Failed to initialize embeddings. Please check your Hugging Face token.")
        return None


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


# Load FAISS index from S3
def load_faiss_index_from_s3(s3_client, embeddings):
    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_PROFORMA_INDEX_PATH)

        if 'Contents' not in response:
            logging.warning("No FAISS index found in S3.")
            return None

        with tempfile.TemporaryDirectory() as temp_dir:
            for obj in response['Contents']:
                key = obj['Key']
                local_path = os.path.join(temp_dir, os.path.basename(key))
                s3_client.download_file(S3_BUCKET, key, local_path)
                logging.info(f"Downloaded FAISS file: {key}")

            vector_store = FAISS.load_local(temp_dir, embeddings, allow_dangerous_deserialization=True)
            logging.info("Successfully loaded FAISS index from S3")
            return vector_store

    except Exception as e:
        logging.error(f"Failed to load FAISS index from S3: {str(e)}")
        return None


# Query the FAISS index
def query_faiss_index(vector_store, query, k=3):
    try:
        results = vector_store.similarity_search(query, k=k)
        return results
    except Exception as e:
        logging.error(f"Error querying FAISS index: {str(e)}")
        return None


# Generate response using Gemini tailored for sales team
def generate_sales_response(model, query, faiss_results):
    try:
        if not faiss_results:
            return "No relevant information found in the proforma invoices for the sales team."

        # Combine FAISS results into a context
        context = "\n\n".join([result.page_content for result in faiss_results])
        prompt = (
            f"You are assisting a sales team. Based on the following information from proforma invoices:\n\n"
            f"{context}\n\n"
            f"Provide a concise, sales-focused answer to the query: {query}"
        )

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logging.error(f"Error generating response with Gemini: {str(e)}")
        return "An error occurred while generating the response."


# Main chatbot interface
def main():
    st.title("Sales Team Proforma Invoice Chatbot")

    # Initialize components
    embeddings = init_embeddings()
    s3_client = init_s3_client()
    gemini_model = init_gemini()
    vector_store = None

    if not all([embeddings, s3_client, gemini_model]):
        return

    # Load FAISS index
    with st.spinner("Loading FAISS index from S3..."):
        vector_store = load_faiss_index_from_s3(s3_client, embeddings)

    if vector_store:
        st.success("FAISS index successfully loaded from S3!")
        st.write(f"FAISS index is loaded from: s3://{S3_BUCKET}/{S3_PROFORMA_INDEX_PATH}")
    else:
        st.error("No FAISS index found in S3 or failed to load the index.")
        return

    # Query input and response
    st.subheader("Sales Team Query")
    query = st.text_input(
        "Enter your sales-related query about the proforma invoices (e.g., 'What is the total amount?', 'Who is the client?'):")

    if st.button("Submit"):
        if query:
            with st.spinner("Searching and generating sales-focused response..."):
                # Search FAISS index
                faiss_results = query_faiss_index(vector_store, query)

                # Generate sales-focused response with Gemini
                response = generate_sales_response(gemini_model, query, faiss_results)

            st.subheader("Sales Team Response")
            st.write(response)
        else:
            st.warning("Please enter a query.")


if __name__ == "__main__":
    main()