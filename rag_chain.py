# %%
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# %%
from config import EMBEDDING_MODEL, VECTORSTORE_DIR

# %%
from dotenv import load_dotenv
import os
load_dotenv()
chat_history=[]

# %%
embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# %%
vector_store = Chroma(
    persist_directory=VECTORSTORE_DIR,
    embedding_function=embedding
)

# %%
retriever = vector_store.as_retriever(search_kwargs={"k":3})

# %%
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    api_key=os.getenv("GROQ_AI_KEY")
)


# %%
from langchain_core.prompts import PromptTemplate

# %%
rag_prompt = PromptTemplate(input_variables=["context", "question"], 
                            template="""
You are a helpful assistant.
Answer ONLY using the context below.
If the answer is not present in the context, say "I don't know".

Context:
{context}

Question:
{question}
""")

# %%
def ask_question(question: str):
    docs = retriever.invoke(question)

    print("\n--- RETRIEVED DOCUMENTS ---\n")
    for i, doc in enumerate(docs, start=1):
        print(f"[Chunk {i}]")
        print(doc.page_content)
        print("Metadata:", doc.metadata)
        print("-" * 40)

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = rag_prompt.format(
        context=context,
        question=question
    )

    response = llm.invoke(prompt)

    sources = set()
    for doc in docs:
        if "source" in doc.metadata:
            sources.add(doc.metadata["source"])

    return response.content, list(sources)


# %%
if __name__ == "__main__":
    answer, sources = ask_question("fill the blanks The Floating ____ ?")

    print("\n--- FINAL ANSWER ---\n")
    print(answer)

    print("\n--- SOURCES ---\n")
    for source in sources:
        print(source)


# %%
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

# %%
def ask_question(question: str):
    global chat_history

    docs = retriever.invoke(question)

    context = "\n\n".join(doc.page_content for doc in docs)

    history_text = "\n".join(
        f"User: {h['user']}\nAssistant: {h['assistant']}"
        for h in chat_history[-3:]
    )

    prompt = rag_prompt.format(
        chat_history=history_text,
        context=context,
        question=question
    )

    response = llm.invoke(prompt)

    chat_history.append({
        "user": question,
        "assistant": response.content
    })

    sources = {doc.metadata.get("source", "") for doc in docs}

    return response.content, list(sources)


# %%



