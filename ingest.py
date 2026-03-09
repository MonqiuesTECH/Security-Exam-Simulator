import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def ingest_pdfs():
    print("Loading PDFs...")
    pdf_files = [
        "data/MesserPracticeExams701.pdf",
        "data/professor-messer-sy0-701-comptia-security-plus-course-notes-v106.pdf"
    ]
    
    docs = []
    for file in pdf_files:
        if os.path.exists(file):
            loader = PyPDFLoader(file)
            docs.extend(loader.load())
            print(f"Successfully loaded {file}")
        else:
            print(f"Warning: {file} not found. Please add it to the data/ folder.")
            
    if not docs:
        print("No documents found. Exiting.")
        return

    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    print("Creating vector database with free Hugging Face embeddings. This may take a minute...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    print("Saving vector database locally to 'faiss_index'...")
    vectorstore.save_local("faiss_index")
    print("Done! You can now run the Streamlit app.")

if __name__ == "__main__":
    ingest_pdfs()
