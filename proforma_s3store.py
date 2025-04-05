import imaplib
import email
import boto3
import os
import logging
import io
import tempfile
from email.header import decode_header
import toml
from PyPDF2 import PdfReader, errors
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
import time
import pdfplumber

# Configuration constants
SECRETS_FILE_PATH = "C:/Users/Admin/.vscode/s3/.streamlit/secrets.toml"
IMAP_SERVER = "imap.gmail.com"
S3_BUCKET = "kalika-rag"
S3_FOLDER = "proforma_invoice/"
S3_PROFORMA_INDEX_PATH = "faiss_indexes/proforma_faiss_index/"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PROCESSED_SUFFIX = "_processed"
 


# Load secrets from secrets.toml
secrets = toml.load(SECRETS_FILE_PATH)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Email and S3 credentials
EMAIL_ACCOUNT = secrets["gmail_uname"]
EMAIL_PASSWORD = secrets["gmail_pwd"]
AWS_ACCESS_KEY = secrets["access_key_id"]
AWS_SECRET_KEY = secrets["secret_access_key"] 

# Initialize S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=secrets["access_key_id"],
    aws_secret_access_key=secrets["secret_access_key"],
)

# Initialize embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False})

def clean_filename(filename):
    """Sanitize filename while preserving original extension if valid."""
    try:
        decoded_name = decode_header(filename)[0][0]
        if isinstance(decoded_name, bytes):
            filename = decoded_name.decode(errors='ignore')
        else:
            filename = str(decoded_name)
    except:
        filename = "proforma_invoice"

    name, ext = os.path.splitext(filename)
    cleaned_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in name)
    return f"{cleaned_name}.pdf" if ext.lower() == '.pdf' else f"{cleaned_name}.pdf"

def is_valid_pdf(content):
    """Verify if content is a valid PDF."""
    try:
        PdfReader(io.BytesIO(content), strict=False) 
        return True
    except (errors.PdfReadError, ValueError, TypeError):
        return False

def file_exists_in_s3(bucket, key):
    """Check if a file exists in S3."""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except s3_client.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        logging.error(f"S3 check error: {e}")
        return False

def upload_to_s3(file_content, bucket, key):
    """Upload file content directly to S3."""
    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=file_content,
            ContentType='application/pdf'
        )
        logging.info(f"Uploaded to S3: {key}")
        return True
    except Exception as e:
        logging.error(f"Upload failed for {key}: {e}")
        return False

def process_pdf_content(file_content):
    """Extract and chunk text from valid PDF bytes using pdfplumber with enhanced error handling."""
    text = ""
    try:
        pdf_stream = io.BytesIO(file_content)
        with pdfplumber.open(pdf_stream) as pdf:
            for page in pdf.pages:
                try:
                    extracted_text = page.extract_text()
                    if extracted_text:
                        text += extracted_text + "\n"
                except Exception as page_err:
                    logging.error(f"Error extracting text from page: {page_err}")
                    continue  # Skip to the next page

    except Exception as e:
        logging.error(f"PDF processing error: {str(e)}")
        return []

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def process_proforma_emails():
    """Process emails with improved PDF validation and error handling."""
    try:
        with imaplib.IMAP4_SSL(IMAP_SERVER) as mail:
            mail.login(EMAIL_ACCOUNT, EMAIL_PASSWORD)
            logging.info("Successfully authenticated with email server")

            mail.select("inbox")
            search_criteria = '(OR SUBJECT "Proforma_Invoice" SUBJECT "proforma_invoice" SUBJECT "proforma pdf" SUBJECT "Proforma")'
            status, email_ids = mail.search(None, search_criteria)

            if status != "OK":
                logging.warning("No matching emails found")
                return

            processed_files = 0
            
            for e_id in email_ids[0].split()[-10:]:  # Process last 10 emails
                try:
                    status, msg_data = mail.fetch(e_id, "(RFC822)")
                    if status != "OK":
                        continue

                    for response_part in msg_data:
                        if isinstance(response_part, tuple):
                            msg = email.message_from_bytes(response_part[1])
                            for part in msg.walk():
                                if part.get_content_maintype() == 'multipart':
                                    continue

                                if part.get_filename() and part.get_content_type() == 'application/pdf':
                                    filename = clean_filename(part.get_filename())
                                    file_content = part.get_payload(decode=True)

                                    if not file_content:
                                        logging.warning(f"Skipping empty attachment: {filename}")
                                        continue

                                    if not is_valid_pdf(file_content):
                                        logging.warning(f"Skipping invalid PDF: {filename}")
                                        continue

                                    key = f"{S3_FOLDER}{filename}"
                                    if not file_exists_in_s3(S3_BUCKET, key):
                                        if upload_to_s3(file_content, S3_BUCKET, key):
                                            processed_files += 1
                                            
                                            
                                    else:
                                        logging.info(f"Skipping existing file: {key}")
                                    processed_files += 1
                except Exception as e:
                    logging.error(f"Error processing email {e_id}: {str(e)}")

            logging.info(f"Processing complete. Uploaded {processed_files} new valid PDFs.")

    except Exception as e:
        logging.error(f"Email processing failed: {str(e)}")
        raise

def merge_existing_indexes(s3_client, bucket, prefix, embeddings, new_chunks):
    """
    Merge new text chunks with existing FAISS indexes stored in S3.
    """
    try:
        # List all objects in the specified S3 prefix
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        
        # If there are no objects in the prefix, return a new FAISS index from the chunks
        if 'Contents' not in response:
            logging.info("No existing FAISS indexes found. Creating a new index.")
            return FAISS.from_texts(new_chunks, embeddings)
        
        # Load existing FAISS indexes and accumulate vectors
        existing_vectors = []
        for obj in response['Contents']:
            if obj['Key'].endswith('.faiss') or obj['Key'].endswith('.pkl'):
                try:
                    # Download the index file from S3 to a temporary location
                    local_path = os.path.join(tempfile.gettempdir(), os.path.basename(obj['Key']))
                    s3_client.download_file(bucket, obj['Key'], local_path)
                    
                    # Load the downloaded FAISS index
                    vector_store = FAISS.load_local(os.path.dirname(local_path), os.path.splitext(os.path.basename(local_path))[0], embeddings,allow_dangerous_deserialization=True)
                    
                    # Extend the existing vectors with the vectors from the loaded index
                    existing_vectors.extend(vector_store.get_texts())
                    logging.info(f"Successfully loaded and merged index from {obj['Key']}")
                except Exception as e:
                    logging.error(f"Error loading index {obj['Key']}: {e}")
                    continue

        # Combine existing vectors with new chunks
        all_chunks = existing_vectors + new_chunks
        
        # Create a new FAISS index from all chunks
        if all_chunks:
            logging.info(f"Creating a new FAISS index with {len(all_chunks)} chunks.")
            return FAISS.from_texts(all_chunks, embeddings)
        else:
            logging.info("No chunks to create FAISS index from.")
            return None

    except Exception as e:
        logging.error(f"Failed to merge existing indexes: {str(e)}")
        return FAISS.from_texts(new_chunks, embeddings)  # Return a new index in case of failure

def create_faiss_index():
    """Create FAISS index from PDFs in S3, merge with existing index, and upload to S3 with timestamp."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_FOLDER)
        
        if 'Contents' not in response:
            logging.info("No PDFs found in S3 proforma folder")
            return

        all_chunks = []
        processed_pdfs = 0

        for obj in response['Contents']:
            key = obj['Key']
            if key.endswith('.pdf'):  # Process all PDFs
                pdf_obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
                pdf_content = pdf_obj['Body'].read()
                chunks = process_pdf_content(pdf_content)
                if chunks:
                    all_chunks.extend(chunks)
                    processed_pdfs += 1
                    logging.info(f"Processed PDF: {key}")

        if not all_chunks:
            logging.info("No valid text chunks extracted from PDFs")
            return

        # Merge new chunks with existing indexes
        vector_store = merge_existing_indexes(s3_client, S3_BUCKET, S3_PROFORMA_INDEX_PATH, embeddings, all_chunks)

        if not vector_store:
            logging.info("No vector store created.")
            return

        # Save FAISS index to a timestamped directory in S3
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = os.path.join(temp_dir, "faiss_index")
            vector_store.save_local(index_path)
            
            # Upload each file in the index directory to S3 with a timestamped prefix
            for file_name in os.listdir(index_path):
                local_file = os.path.join(index_path, file_name)
                s3_key = f"{S3_PROFORMA_INDEX_PATH}{file_name}"
                
                with open(local_file, 'rb') as f:
                    s3_client.put_object(
                        Bucket=S3_BUCKET,
                        Key=s3_key,
                        Body=f
                    )
                    logging.info(f"Uploaded FAISS index file: {s3_key}")

        logging.info(f"FAISS index creation completed successfully. "
                     f"Processed {processed_pdfs} PDFs. "
                     f"Uploaded index to S3 at {S3_PROFORMA_INDEX_PATH}")

    except Exception as e:
        logging.error(f"FAISS index creation failed: {str(e)}")

if __name__ == "__main__":
    process_proforma_emails()
    create_faiss_index()
