import os
import streamlit as st
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import logging
import boto3
from botocore.exceptions import ClientError
import tempfile

# --- Logging Setup ---
logging.basicConfig(
    filename='sales_rag.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("sales_rag")

# --- Configuration ---
CONFIG = {
    "embeddings_model": "sentence-transformers/all-MiniLM-L6-v2",
    "faiss_paths": {
        "po": "s3://kalika-rag/faiss_indexes/po_faiss_index/",  # Full S3 path to PO FAISS index
        "proforma": "s3://kalika-rag/faiss_indexes/proforma_faiss_index/"  # Full S3 path to Proforma FAISS index
    },
    "google_model": st.secrets["GEMINI_MODEL"],  # Gemini model name from secrets
    "query_enhancements": {
        "po": ["include shipment terms", "reference PO number", "note delivery address"],
        "proforma": ["include pricing terms", "reference payment conditions", "note delivery timelines"]
    }
}

# --- Document Type Detection ---
def detect_document_type(query):
    """Route queries to appropriate FAISS index"""
    po_keywords = {'po', 'purchase order', 'shipment'}
    proforma_keywords = {'proforma', 'invoice', 'payment'}

    query_lower = query.lower()
    if any(kw in query_lower for kw in po_keywords):
        return 'po'
    elif any(kw in query_lower for kw in proforma_keywords):
        return 'proforma'
    return 'general'

# --- Enhanced Prompt Engineering ---
SALES_PROMPT_TEMPLATE = """Analyze this {doc_type} document:
{context}

Question: {question}

Format requirements:
- Currency amounts in USD
- {date_requirement}
- Reference {clause_type} numbers
- Add "Verify with original document for binding terms" disclaimer"""

PROMPT_CONFIG = {
    "po": {
        "date_requirement": "Highlight shipment dates in bold",
        "clause_type": "PO clause"
    },
    "proforma": {
        "date_requirement": "Highlight expiration dates in bold",
        "clause_type": "invoice clause"
    }
}

# --- RAG System Core ---
class SalesRAGSystem:
    def __init__(self):
        self.s3_client = self._initialize_s3_client()
        self.embeddings = HuggingFaceEmbeddings(model_name=CONFIG["embeddings_model"])
        self.vector_stores = self._load_vector_stores()
        self.llm = ChatGoogleGenerativeAI(
            model=CONFIG["google_model"],
            temperature=0.2,
            google_api_key=st.secrets["gemini_api_key"]  # Gemini API key from secrets
        )

    def _initialize_s3_client(self):
        """Create authenticated S3 client using credentials from Streamlit secrets"""
        return boto3.client(
            's3',
            aws_access_key_id=st.secrets["access_key_id"],
            aws_secret_access_key=st.secrets["secret_access_key"],
            region_name=st.secrets["AWS_REGION"]
        )

    def _download_s3_folder(self, s3_path, local_dir):
        """Download entire FAISS index folder from S3"""
        bucket_name = s3_path.split('/')[2]
        prefix = '/'.join(s3_path.split('/')[3:])
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
                for obj in page.get('Contents', []):
                    s3_key = obj['Key']
                    local_path = os.path.join(local_dir, os.path.relpath(s3_key, prefix))
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    self.s3_client.download_file(bucket_name, s3_key, local_path)
        except ClientError as e:
            logger.error(f"S3 download failed: {str(e)}")
            raise RuntimeError("Failed to download FAISS index from S3.")

    def _load_vector_stores(self):
        """Load indexes directly from S3 paths"""
        stores = {}
        for doc_type, s3_path in CONFIG["faiss_paths"].items():
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    self._download_s3_folder(s3_path, temp_dir)
                    stores[doc_type] = FAISS.load_local(
                        temp_dir,
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    logger.info(f"Loaded {doc_type} index from S3 path: {s3_path}")
            except Exception as e:
                logger.error(f"Failed loading {doc_type}: {str(e)}")
                raise RuntimeError(f"Failed to load FAISS index for {doc_type}.")
        return stores

    def query_transform(self, user_query, doc_type):
        """Enhance query with document-specific context"""
        enhancements = CONFIG["query_enhancements"].get(doc_type, [])
        return f"{user_query} [Context: {' '.join(enhancements)}]"

    def create_pipeline(self, doc_type):
        """Create retrieval pipeline for specific document type"""
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vector_stores[doc_type].as_retriever(search_kwargs={"k": 5}),
            chain_type="map_reduce",
            return_source_documents=False,  # Don't include source documents in the output
            chain_type_kwargs={
                "prompt": self._create_prompt(doc_type),
                "document_prompt": PromptTemplate(
                    input_variables=["page_content"],
                    template="{page_content}"
                )
            }
        )

    def _create_prompt(self, doc_type):
        """Generate document-specific prompt"""
        return PromptTemplate(
            template=SALES_PROMPT_TEMPLATE,
            input_variables=["context", "question"],
            partial_variables={
                "doc_type": doc_type.upper(),
                **PROMPT_CONFIG.get(doc_type, {})
            }
        )

# --- Streamlit Interface ---
def main():
    st.title("Sales Document Chatbot")

    query = st.text_input("Ask about documents:",
                          placeholder="e.g., 'Payment terms for INV-2024-789'")

    if query:
        try:
            rag_system = SalesRAGSystem()

            doc_type = detect_document_type(query)
            processed_query = rag_system.query_transform(query, doc_type)

            pipeline = rag_system.create_pipeline(doc_type)
            result = pipeline.invoke({"query": processed_query})

            # Display chatbot response
            st.subheader(f"{doc_type.upper()} Response")
            st.markdown(result["result"])

            # Audit logging
            logger.info(f"""
                Document Type: {doc_type}
                Original Query: {query}
                Enhanced Query: {processed_query}
                Response: {result['result'][:300]}...
            """)

        except Exception as e:
            st.error("Analysis failed. Please try again.")
            logger.error(f"Query failure: {str(e)}")

if __name__ == "__main__":
    main()
