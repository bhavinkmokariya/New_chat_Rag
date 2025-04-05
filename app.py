import os
import logging
import boto3
import tempfile
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
from typing import List, Optional
import streamlit as st

# Configuration constants
S3_BUCKET = "kalika-rag"
PROFORMA_FOLDER = "proforma_invoice/"
PO_FOLDER = "PO_Dump/"
PROFORMA_INDEX_PATH = "faiss_indexes/proforma_faiss_index/"
PO_INDEX_PATH = "faiss_indexes/po_faiss_index/"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_API_KEY = "AIzaSyBdE-BuXNESWGaXEQDZ5gJxgRqlyzchltM"  # Replace with your actual API key

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)


class SalesChatbot:
    def __init__(self):
        # Configure Gemini API
        genai.configure(api_key=GEMINI_API_KEY)
        self.gemini_model = genai.GenerativeModel('gemini-1.5-pro')

        # Initialize S3 client
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=st.secrets["access_key_id"],
            aws_secret_access_key=st.secrets["secret_access_key"],
        )

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )

        # Load indices
        self.proforma_index = self.load_faiss_index(PROFORMA_INDEX_PATH)
        self.po_index = self.load_faiss_index(PO_INDEX_PATH)

        # Log initialization status
        if self.proforma_index:
            logging.info("Proforma index loaded successfully")
        else:
            logging.warning("Failed to load proforma index")

        if self.po_index:
            logging.info("PO index loaded successfully")
        else:
            logging.warning("Failed to load PO index")

    def load_faiss_index(self, s3_prefix: str) -> Optional[FAISS]:
        """Load FAISS index from S3 with proper file handling."""
        try:
            # List all objects in the prefix
            response = self.s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=s3_prefix)

            if 'Contents' not in response:
                logging.warning(f"No files found in S3 at {s3_prefix}")
                return None

            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                logging.info(f"Created temporary directory: {temp_dir}")

                # Download all .faiss and .pkl files
                faiss_file = None
                pkl_file = None

                for obj in response['Contents']:
                    key = obj['Key']
                    filename = os.path.basename(key)
                    local_path = os.path.join(temp_dir, filename)

                    # Only download .faiss and .pkl files
                    if filename.endswith('.faiss') or filename.endswith('.pkl'):
                        self.s3_client.download_file(S3_BUCKET, key, local_path)
                        logging.info(f"Downloaded {key} to {local_path}")

                        if filename.endswith('.faiss'):
                            faiss_file = filename
                        elif filename.endswith('.pkl'):
                            pkl_file = filename

                if not faiss_file or not pkl_file:
                    logging.error(f"Missing required FAISS index files in {s3_prefix}")
                    return None

                # Create simple index files with standard names
                index_name = "faiss_index"

                # Rename files to standard names expected by FAISS
                os.rename(
                    os.path.join(temp_dir, faiss_file),
                    os.path.join(temp_dir, f"{index_name}.faiss")
                )
                os.rename(
                    os.path.join(temp_dir, pkl_file),
                    os.path.join(temp_dir, f"{index_name}.pkl")
                )

                logging.info(f"Renamed files to standard format in {temp_dir}")

                # List directory contents for debugging
                logging.info(f"Directory contents: {os.listdir(temp_dir)}")

                # Load the index with the standard name
                index = FAISS.load_local(temp_dir, index_name, self.embeddings, allow_dangerous_deserialization=True)
                logging.info(f"Successfully loaded FAISS index from {s3_prefix}")
                return index

        except Exception as e:
            logging.error(f"Failed to load FAISS index from {s3_prefix}: {str(e)}")
            logging.exception("Detailed error:")
            return None

    def query_gemini(self, prompt: str) -> str:
        """Call Gemini 1.5 Pro to generate a response."""
        try:
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=500,
                    temperature=0.2
                )
            )
            return response.text
        except Exception as e:
            logging.error(f"Gemini API error: {str(e)}")
            return "I'm having trouble connecting to my AI brain right now. Please try again later."

    def search_documents(self, query: str, index_type: str) -> List[str]:
        """Search FAISS index for relevant document chunks."""
        index = self.proforma_index if index_type == "proforma" else self.po_index

        if not index:
            return [f"I don't have access to the {index_type} documents right now."]

        try:
            results = index.similarity_search(query, k=3)  # Top 3 matches
            return [doc.page_content for doc in results]
        except Exception as e:
            logging.error(f"Error searching {index_type} index: {str(e)}")
            return ["Error during document search."]

    def process_query(self, user_input: str) -> str:
        """Process user query and return a response."""
        user_input = user_input.lower()

        # Handle missing indices case
        if not self.proforma_index and not self.po_index:
            return ("I'm currently unable to access any document databases. Please check that the FAISS indices "
                    "have been properly uploaded to S3.")

        # Determine intent and document type
        if "proforma" in user_input or "invoice" in user_input:
            doc_type = "proforma"
            if not self.proforma_index:
                return "I'm sorry, but I don't have access to proforma invoice data right now."
            search_results = self.search_documents(user_input, "proforma")
        elif "po" in user_input or "purchase order" in user_input:
            doc_type = "po"
            if not self.po_index:
                return "I'm sorry, but I don't have access to purchase order data right now."
            search_results = self.search_documents(user_input, "po")
        else:
            # Use Gemini to interpret ambiguous queries
            prompt = f"Determine if the following query is about Proforma Invoices or Purchase Orders: '{user_input}'. Respond with only 'proforma' or 'po'."
            doc_type_response = self.query_gemini(prompt).strip().lower()
            doc_type = "proforma" if "proforma" in doc_type_response else "po"

            # Check if we have the required index
            if (doc_type == "proforma" and not self.proforma_index) or (doc_type == "po" and not self.po_index):
                return f"I've determined your question is about {doc_type}, but I don't have access to that data right now."

            search_results = self.search_documents(user_input, doc_type)

        # Construct a prompt for Gemini to generate a natural response
        context = "\n".join(search_results)
        prompt = (
            f"You are a helpful assistant for a sales team. Based on the following document excerpts, "
            f"answer the query: '{user_input}'. If the information is insufficient, say so clearly.\n\n"
            f"Document excerpts:\n{context}"
        )
        response = self.query_gemini(prompt)
        return response


def main():
    """Main function to run the chatbot"""
    print("Initializing Sales Chatbot...")
    try:
        chatbot = SalesChatbot()
        print("Sales Chatbot: How can I assist you today? (Type 'exit' to quit)")

        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue

                if user_input.lower() in ("exit", "quit", "bye"):
                    print("Sales Chatbot: Goodbye!")
                    break

                response = chatbot.process_query(user_input)
                print(f"Sales Chatbot: {response}")

            except EOFError:
                print("\nInput terminated. Exiting...")
                break
            except KeyboardInterrupt:
                print("\nSales Chatbot: Session terminated by user. Goodbye!")
                break
            except Exception as e:
                logging.error(f"Error in chat loop: {str(e)}")
                print("Sales Chatbot: I encountered an error. Please try again.")

    except Exception as e:
        logging.error(f"Failed to initialize chatbot: {str(e)}")
        print("Failed to initialize the Sales Chatbot. Check logs for details.")


if __name__ == "__main__":
    main()