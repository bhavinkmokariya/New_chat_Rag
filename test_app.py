import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load the FAISS index from local storage
faiss_index_path = "C:/Users/shrut/OneDrive/Desktop/index.faiss"  # Replace with the actual path to your FAISS index
retriever = FAISS.load_local(faiss_index_path)

# Set up environment variable for Google Gemini API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyD2m7P6ZT1I-x4rQoFPXxvyNrmSP23eWOI"  # Replace with your actual API key

# Initialize the Gemini 1.5 Pro model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)

# Set up the RAG pipeline using LangChain's RetrievalQA chain
rag_pipeline = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# Example query
query = "What are the impacts of climate change on agriculture?"
response = rag_pipeline.run(query)

print("Response:", response)
