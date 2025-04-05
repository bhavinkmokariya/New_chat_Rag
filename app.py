import os
import logging
import boto3
import tempfile
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import toml
import google.generativeai as genai
from typing import List, Optional
import streamlit as st

# Configuration constants
#SECRETS_FILE_PATH = "C:/Users/Admin/.vscode/s3/.streamlit/secrets.toml"
S3_BUCKET = "kalika-rag"
PROFORMA_FOLDER = "proforma_invoice/"
PO_FOLDER = "PO_Dump/"
PROFORMA_INDEX_PATH = "faiss_indexes/proforma_faiss_index/"
PO_INDEX_PATH = "faiss_indexes/po_faiss_index/"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_API_KEY = "AIzaSyBdE-BuXNESWGaXEQDZ5gJxgRqlyzchltM"  # Replace with your actual Gemini API key

# Load secrets
#secrets = toml.load(SECRETS_FILE_PATH)

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Initialize S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=st.secrets["access_key_id"],
    aws_secret_access_key=st.secrets["secret_access_key"],
)

# Initialize embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)


class SalesChatbot:
    def __init__(self):
        self.proforma_index = self.load_faiss_index(PROFORMA_INDEX_PATH)
        self.po_index = self.load_faiss_index(PO_INDEX_PATH)
        # Initialize Gemini 1.5 Pro model
        self.gemini_model = genai.GenerativeModel('gemini-1.5-pro')

    def load_faiss_index(self, s3_prefix: str) -> Optional[FAISS]:
        """Load the latest FAISS index from S3."""
        try:
            response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=s3_prefix)
            if 'Contents' not in response:
                logging.warning(f"No FAISS index found at {s3_prefix}")
                return None

            # Download the latest index files
            with tempfile.TemporaryDirectory() as temp_dir:
                for obj in response['Contents']:
                    if obj['Key'].endswith('.faiss') or obj['Key'].endswith('.pkl'):
                        local_path = os.path.join(temp_dir, os.path.basename(obj['Key']))
                        s3_client.download_file(S3_BUCKET, obj['Key'], local_path)

                # Load FAISS index
                index = FAISS.load_local(temp_dir, "faiss_index", embeddings, allow_dangerous_deserialization=True)
                logging.info(f"Loaded FAISS index from {s3_prefix}")
                return index
        except Exception as e:
            logging.error(f"Failed to load FAISS index from {s3_prefix}: {str(e)}")
            return None

    def query_gemini(self, prompt: str) -> str:
        """Call Gemini 1.5 Pro to generate a response."""
        try:
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=500
                )
            )
            return response.text
        except Exception as e:
            logging.error(f"Gemini API error: {str(e)}")
            return "Error: Could not connect to Gemini API."

    def search_documents(self, query: str, index_type: str) -> List[str]:
        """Search FAISS index for relevant document chunks."""
        index = self.proforma_index if index_type == "proforma" else self.po_index
        if not index:
            return [f"No {index_type} index available."]

        try:
            results = index.similarity_search(query, k=3)  # Top 3 matches
            return [doc.page_content for doc in results]
        except Exception as e:
            logging.error(f"Error searching {index_type} index: {str(e)}")
            return ["Error during search."]

    def process_query(self, user_input: str) -> str:
        """Process user query and return a response."""
        user_input = user_input.lower()

        # Determine intent and document type
        if "proforma" in user_input or "invoice" in user_input:
            doc_type = "proforma"
            search_results = self.search_documents(user_input, "proforma")
        elif "po" in user_input or "purchase order" in user_input:
            doc_type = "po"
            search_results = self.search_documents(user_input, "po")
        else:
            # Use Gemini to interpret ambiguous queries
            prompt = f"Determine if the following query is about Proforma Invoices or Purchase Orders: '{user_input}'. Respond with 'proforma' or 'po'."
            doc_type = self.query_gemini(prompt).strip()
            search_results = self.search_documents(user_input, doc_type)

        # Construct a prompt for Gemini to generate a natural response
        context = "\n".join(search_results)
        prompt = (
            f"You are a helpful assistant for a sales team. Based on the following document excerpts, "
            f"answer the query: '{user_input}'. If the information is insufficient, say so.\n\n"
            f"Document excerpts:\n{context}"
        )
        response = self.query_gemini(prompt)
        return response


def main():
    chatbot = SalesChatbot()
    print("Sales Chatbot: How can I assist you today? (Type 'exit' to quit)")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Sales Chatbot: Goodbye.")
            break

        response = chatbot.process_query(user_input)
        print(f"Sales Chatbot: {response}")


if __name__ == "__main__":
    main()