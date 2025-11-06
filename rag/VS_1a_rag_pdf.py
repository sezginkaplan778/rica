import os


from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from torch.ao.nn.quantized.functional import threshold

load_dotenv()




# --- Define directories ---
current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, "books")  # PDF file instead of .txt
persistent_directory = os.path.join(current_dir, "db", "chroma_pdf_db")

# --- Initialize embeddings ---
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# --- Check if persistent Chroma database exists ---
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # Ensure the PDF file exists
    if not os.path.exists(books_dir):
        raise FileNotFoundError(f"The file {books_dir} does not exist. Please check the path.")

    # --- Load PDF file ---
    #loader = PyPDFLoader(file_path)
    #documents = loader.load()
    # List all text files in the directory
    book_files = [f for f in os.listdir(books_dir) if f.endswith(".pdf")]
    # Read the text content from each file and store it with metadata
    documents = []
    for book_file in book_files:
        file_path = os.path.join(books_dir, book_file)
        loader = PyPDFLoader(file_path)
        book_docs = loader.load()
        for doc in book_docs:
            # Add metadata to each document indicating its source
            doc.metadata = {"source": book_file}
            documents.append(doc)





    # --- Split the PDF into chunks ---
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Add metadata for traceability
    #for i, doc in enumerate(docs):
    #   doc.metadata = {"source": "CV.pdf", "chunk_id": i}



    # --- Create and persist vector store ---
    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
    db.persist()
    print("✅ Vector store created and persisted successfully.")
else:
    print("Vector store already exists. Loading existing database...")
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
    print("✅ Vector store loaded successfully.")

# --- Optional test query ---
query = "What are certificate programs in the Richmond College?"
#results = db.similarity_search(query,k=1)
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10},
)
relevant_docs = retriever.invoke(query)
# Display the relevant results with metadata
print("\n--- Retrieval Results ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    print(f"Source: {doc.metadata.get('source')}\n")

