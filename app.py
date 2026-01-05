import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate


load_dotenv()

from config import EMBEDDING_MODEL, VECTORSTORE_DIR

st.set_page_config(
    page_title="RAG Chatbot",
    layout="centered"
)

st.title("📄 RAG Document Chatbot")
st.caption("Ask questions based on your documents")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

vector_store = Chroma(
    persist_directory=VECTORSTORE_DIR,
    embedding_function=embeddings
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2
    # api_key=os.getenv("GROQ_API_KEY")
)
rag_prompt = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template="""
You are a helpful assistant.
Use ONLY the context below to answer.
If the answer is not in the context, say "I don't know".

Conversation so far:
{chat_history}

Context:
{context}

Question:
{question}
"""
)
def ask_question(question: str):
    docs = retriever.invoke(question)

    context = "\n\n".join(doc.page_content for doc in docs)

    history_text = "\n".join(
        f"User: {turn['user']}\nAssistant: {turn['assistant']}"
        for turn in st.session_state.chat_history[-3:]
    )

    prompt = rag_prompt.format(
        chat_history=history_text,
        context=context,
        question=question
    )

    response = llm.invoke(prompt)

    st.session_state.chat_history.append({
        "user": question,
        "assistant": response.content
    })

    sources = {doc.metadata.get("source", "") for doc in docs}

    return response.content, list(sources)

for turn in st.session_state.chat_history:
    st.chat_message("user").write(turn["user"])
    st.chat_message("assistant").write(turn["assistant"])


user_input = st.chat_input("Ask a question about your documents")
if user_input:
    st.chat_message("user").write(user_input)

    with st.spinner("Thinking..."):
        answer, sources = ask_question(user_input)

    st.chat_message("assistant").write(answer)

    if sources:
        with st.expander("Sources"):
            for src in sources:
                st.write(src)
