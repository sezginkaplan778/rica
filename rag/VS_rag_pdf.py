import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

# --- Load environment variables ---
load_dotenv()

# --- Define directories ---
current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, "books")
persistent_directory = os.path.join(current_dir, "db", "chroma_pdf_db")

# --- Initialize embedding model ---
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
#embeddings = OllamaEmbeddings(model="nomic-embed-text")
# --- Load PDFs (always needed for BM25) ---
documents = []
for book_file in os.listdir(books_dir):
    if book_file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(books_dir, book_file))
        book_docs = loader.load()
        for doc in book_docs:
            doc.metadata["source"] = book_file
        documents.extend(book_docs)

# --- Split PDFs into chunks (for both Chroma and BM25) ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""]
)
docs = text_splitter.split_documents(documents)

# --- Create or load Chroma DB ---
if not os.path.exists(persistent_directory) or not os.listdir(persistent_directory):
    print("📚 No existing Chroma DB found. Creating from PDFs...")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
    db.persist()
    print("✅ Chroma vector store created and saved.")
else:
    print("💾 Loading existing Chroma DB...")
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
    print("✅ Chroma vector store loaded successfully.")

# --- Create BM25 (keyword) retriever ---
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 8

# --- Create Chroma (semantic) retriever ---
chroma_retriever = db.as_retriever(search_kwargs={"k": 8})

# --- Combine both in a Hybrid Retriever ---
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, chroma_retriever],
    weights=[0.5, 0.5]
)

print("🤝 Hybrid retriever initialized successfully!")

# --- Test the hybrid retrieval ---
query = "What are the certificate programs offered by Richmond College?"
results = hybrid_retriever.invoke(query)

print("\n--- Hybrid Retrieval Results ---")
for i, doc in enumerate(results, 1):
    print(f"Document {i} (source: {doc.metadata.get('source', 'unknown')}):")
    print(doc.page_content[:400])
    print("-" * 80)



