import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict
from fastapi.middleware.cors import CORSMiddleware

from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from langchain.memory import ConversationBufferMemory
#from langserve import add_routes

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
     "Be clear, supportive, and factual."
     "Your only source of truth is the retrieved documents provided to you as context. "
    """Follow these rules strictly:

    1. You must base all answers *only* on the retrieved content in the current context.
    2. You are not allowed to use your own world knowledge, training data, or make assumptions.
    3. Do not browse the web, search online, or generate speculative information.
    4. If the answer is not present or cannot be inferred directly from the context, clearly state:
        "I'm not able to answer this question at the moment. Please ask another question about Richmond College."
    5. When referencing or summarizing, accurately represent the information in the documents — 
        do not paraphrase in ways that add new facts or interpretations.
    6. If multiple documents contain conflicting information, summarize the differences factually.
    7. If the input is a greeting prompt, give a gentle response and ask the question related to Richmond College"""
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
    message: str
    class Config:
        arbitrary_types_allowed = True

class ChatOutput(BaseModel):
    answer: str

# === FastAPI App ===
app = FastAPI(title="RICA Assistant", version="1.0", description="Richmond College AI Assistant")

# === Allow browser connection (CORS) ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Simple chat endpoint for the HTML frontend ===
@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    message = data.get("message", "").strip()
    if not message:
        return {"response": "Please provide a message."}
    try:
        result = conversational_rag(message)
        return {"response": result["answer"]}
    except Exception as e:
        return {"response": f"Error: {str(e)}"}

# === Add LangServe Route ===


if __name__ == "__main__":
    import uvicorn

    # Render automatically assigns a PORT environment variable
    import os
    port = int(os.environ.get("PORT", 8000))

    print(f"🚀 Starting RICA backend on port {port}")
    print("💡 When deployed on Render, it will be available at https://rica-v45l.onrender.com")

    uvicorn.run(app, host="0.0.0.0", port=port)

    
