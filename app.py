import os
import logging
import boto3
import tempfile
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
from typing import List, Optional
import streamlit as st
import time

# Configuration constants
S3_BUCKET = "kalika-rag"
PROFORMA_FOLDER = "proforma_invoice/"
PO_FOLDER = "PO_Dump/"
PROFORMA_INDEX_PATH = "faiss_indexes/proforma_faiss_index/"
PO_INDEX_PATH = "faiss_indexes/po_faiss_index/"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_RETRIES = 3
RETRY_DELAY = 3  # seconds

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)


class SalesChatbot:
    def __init__(self):
        # Initialize Gemini model first
        self.gemini_api_key = st.secrets.get("gemini_api_key", "")
        if not self.gemini_api_key:
            logging.warning("Gemini API key not found in secrets. Some features may not work.")
        else:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-pro')

        # Initialize S3 client
        try:
            self.s3_client = boto3.client(
                "s3",
                aws_access_key_id=st.secrets["access_key_id"],
                aws_secret_access_key=st.secrets["secret_access_key"],
            )
        except Exception as e:
            logging.error(f"Failed to initialize S3 client: {str(e)}")
            self.s3_client = None

        # Initialize embeddings
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': False}
            )
        except Exception as e:
            logging.error(f"Failed to initialize embeddings model: {str(e)}")
            self.embeddings = None

        # Load indices if possible
        self.proforma_index = None
        self.po_index = None

        if self.s3_client and self.embeddings:
            self.proforma_index = self.load_faiss_index(PROFORMA_INDEX_PATH)
            self.po_index = self.load_faiss_index(PO_INDEX_PATH)

    def load_faiss_index(self, s3_prefix: str) -> Optional[FAISS]:
        """Load FAISS index from S3 with retries."""
        if not self.s3_client:
            return None

        for attempt in range(MAX_RETRIES):
            try:
                response = self.s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=s3_prefix)

                if 'Contents' not in response:
                    logging.warning(f"No FAISS index found at {s3_prefix}")
                    return None

                # Create temp directory with unique name to avoid conflicts
                with tempfile.TemporaryDirectory(prefix=f"faiss_index_{s3_prefix.replace('/', '_')}") as temp_dir:
                    # Track what we downloaded to ensure we have all required files
                    downloaded_files = []

                    for obj in response['Contents']:
                        if obj['Key'].endswith('.faiss') or obj['Key'].endswith('.pkl'):
                            filename = os.path.basename(obj['Key'])
                            local_path = os.path.join(temp_dir, filename)
                            self.s3_client.download_file(S3_BUCKET, obj['Key'], local_path)
                            downloaded_files.append(filename)

                    # Verify we have both required file types
                    has_faiss = any(f.endswith('.faiss') for f in downloaded_files)
                    has_pkl = any(f.endswith('.pkl') for f in downloaded_files)

                    if not (has_faiss and has_pkl):
                        logging.error(f"Missing required index files in {s3_prefix}. Found: {downloaded_files}")
                        return None

                    # Load the index using the correct index name (the part before the file extension)
                    index_name = None
                    for f in downloaded_files:
                        if f.endswith('.faiss'):
                            index_name = f[:-6]  # Remove '.faiss' extension
                            break

                    if not index_name:
                        logging.error(f"Could not determine index name from files: {downloaded_files}")
                        return None

                    index = FAISS.load_local(temp_dir, index_name, self.embeddings,
                                             allow_dangerous_deserialization=True)
                    logging.info(f"Successfully loaded FAISS index from {s3_prefix}")
                    return index

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    logging.warning(
                        f"Attempt {attempt + 1}/{MAX_RETRIES} failed to load index from {s3_prefix}: {str(e)}")
                    time.sleep(RETRY_DELAY)
                else:
                    logging.error(f"All attempts failed to load FAISS index from {s3_prefix}: {str(e)}")

        return None

    def query_gemini(self, prompt: str) -> str:
        """Call Gemini 1.5 Pro to generate a response."""
        if not hasattr(self, 'gemini_model'):
            return "Gemini API is not configured properly."

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
            return "I'm having trouble connecting to my knowledge base right now. Could you please try again later or ask a different question?"

    def search_documents(self, query: str, index_type: str) -> List[str]:
        """Search FAISS index for relevant document chunks."""
        index = self.proforma_index if index_type == "proforma" else self.po_index

        if not index:
            return [
                f"I don't have access to the {index_type} database right now. Please check if the indices have been properly loaded."]

        try:
            results = index.similarity_search(query, k=3)  # Top 3 matches
            return [doc.page_content for doc in results]
        except Exception as e:
            logging.error(f"Error searching {index_type} index: {str(e)}")
            return ["I'm having trouble searching through the documents right now."]

    def process_query(self, user_input: str) -> str:
        """Process user query and return a response."""
        if not user_input.strip():
            return "Please ask me a question about proforma invoices or purchase orders."

        user_input = user_input.lower()

        # Check if indices are available
        if not (self.proforma_index or self.po_index):
            return ("I'm currently unable to access the document database. Please check that the FAISS indices "
                    "have been properly uploaded to S3 and that your AWS credentials are correct.")

        # Determine intent and document type
        if "proforma" in user_input or "invoice" in user_input:
            doc_type = "proforma"
            search_results = self.search_documents(user_input, "proforma")
        elif "po" in user_input or "purchase order" in user_input:
            doc_type = "po"
            search_results = self.search_documents(user_input, "po")
        else:
            # Use Gemini to interpret ambiguous queries
            prompt = f"Determine if the following query is about Proforma Invoices or Purchase Orders: '{user_input}'. Respond with only 'proforma' or 'po'."
            doc_type_response = self.query_gemini(prompt).strip().lower()
            doc_type = "proforma" if "proforma" in doc_type_response else "po"
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


def run_chatbot():
    """Run the chatbot in console mode"""
    print("Initializing Sales Chatbot...")
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

        except KeyboardInterrupt:
            print("\nSales Chatbot: Session terminated by user. Goodbye!")
            break
        except Exception as e:
            logging.error(f"Unexpected error in main loop: {str(e)}")
            print("Sales Chatbot: I encountered an unexpected error. Please try again.")


def streamlit_app():
    """Run the chatbot as a Streamlit app"""
    st.title("Sales Document Assistant")
    st.write("Ask me questions about your proforma invoices and purchase orders!")

    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = SalesChatbot()

    # Chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    user_input = st.chat_input("Ask something about your invoices or purchase orders...")

    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.process_query(user_input)
                st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    import sys

    # Check if we're running in Streamlit
    if 'streamlit' in sys.modules:
        streamlit_app()
    else:
        run_chatbot()