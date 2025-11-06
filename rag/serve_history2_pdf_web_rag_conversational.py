import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse

from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from langchain.memory import ConversationBufferMemory
from langserve import add_routes

# === Load environment variables ===
load_dotenv()

# === Directories for vector stores ===
current_dir = os.path.dirname(os.path.abspath(__file__))
pdf_dir = os.path.join(current_dir, "db", "chroma_pdf_db")
web_dir = os.path.join(current_dir, "db", "chroma_RC_soup_db")
os.makedirs(pdf_dir, exist_ok=True)
os.makedirs(web_dir, exist_ok=True)

# === Embeddings and Vector Stores ===
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
pdf_db = Chroma(persist_directory=pdf_dir, embedding_function=embeddings)
web_db = Chroma(persist_directory=web_dir, embedding_function=embeddings)

# === Combined Retriever ===
pdf_retriever = pdf_db.as_retriever(search_kwargs={"k": 3})
web_retriever = web_db.as_retriever(search_kwargs={"k": 3})
combined_retriever = EnsembleRetriever(
    retrievers=[pdf_retriever, web_retriever],
    weights=[0.5, 0.5]
)

# === LLM ===
llm = ChatOpenAI(model="gpt-4o-mini")

# === Contextualization Prompt ===
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Given a chat history and the latest user question which might reference context, "
     "reformulate it into a standalone question. Do not answer it."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# === History-Aware Retriever ===
history_retriever = create_history_aware_retriever(
    llm, combined_retriever, contextualize_prompt
)

# === QA Prompt ===
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

# === Create RAG Chain ===
qa_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_retriever, qa_chain)

# === Memory (Keep chat history internally) ===
memory = ConversationBufferMemory(return_messages=True)

def conversational_rag(input_text: str):
    """Run RAG chain with internal chat memory."""
    # Build input with saved chat history
    inputs = {
        "input": input_text,
        "chat_history": memory.chat_memory.messages
    }
    response = rag_chain.invoke(inputs)
    answer = response["answer"]

    # Save new turn to memory
    memory.chat_memory.add_user_message(input_text)
    memory.chat_memory.add_ai_message(answer)
    return {"answer": answer}

# === Input / Output Schemas ===
class ChatInput(BaseModel):
    input: str  # only input text (no chat_history shown!)

class ChatOutput(BaseModel):
    answer: str

# === FastAPI App ===
app = FastAPI(title="RICA Assistant", version="1.0", description="Richmond College AI Assistant")

# === Add LangServe Route ===
add_routes(
    app,
    RunnableLambda(lambda x: conversational_rag(x["input"])["answer"]).with_types(
        input_type=ChatInput,
        output_type=str
    ),
    path="/chain",
)

# === Run App ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)


