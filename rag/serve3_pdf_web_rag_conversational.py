import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.retrievers import EnsembleRetriever
from langserve import add_routes
from langchain.chains import ConversationalRetrievalChain
from langchain.schema.runnable import RunnableLambda
# Load environment variables
load_dotenv()

# Directories for vector stores
current_dir = os.path.dirname(os.path.abspath(__file__))
pdf_dir = os.path.join(current_dir, "db", "chroma_pdf_db")
web_dir = os.path.join(current_dir, "db", "chroma_RC_soup_db")

# Embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Load vector stores
pdf_db = Chroma(persist_directory=pdf_dir, embedding_function=embeddings)
web_db = Chroma(persist_directory=web_dir, embedding_function=embeddings)

# Create retrievers
pdf_retriever = pdf_db.as_retriever(search_kwargs={"k": 3})
web_retriever = web_db.as_retriever(search_kwargs={"k": 3})
combined_retriever = EnsembleRetriever(
    retrievers=[pdf_retriever, web_retriever],
    weights=[0.5, 0.5]
)

# LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Contextualization prompt
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Given a chat history and the latest user question which might reference context, "
     "reformulate it into a standalone question. Do not answer it."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# Create history-aware retriever
history_retriever = create_history_aware_retriever(
    llm, combined_retriever, contextualize_prompt
)

# QA prompt
qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful academic assistant guiding students about Richmond College in the UK. "
     "Be clear, supportive, and factual. "
     "Use the retrieved context to answer and cite the source when possible. "
     "If unsure, say so. "
     "Answer in no more than three sentences.\n\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# Chain setup
qa_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_retriever, qa_chain)

# ✅ Wrap to extract only "answer"
clean_chain = rag_chain | RunnableLambda(lambda x: x["answer"])

# === Input / Output Schemas ===
class ChatInput(BaseModel):
    input: str
    chat_history: list = []

class ChatOutput(BaseModel):
    answer: str

# === FastAPI App ===
app = FastAPI(title="RICA Assistant", version="1.0", description="Richmond College AI Assistant")

# === Add Route ===
add_routes(
    app,
    clean_chain.with_types(input_type=ChatInput, output_type=str),
    path="/chain",
)

# Run app
if __name__ == "__main__":

    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)