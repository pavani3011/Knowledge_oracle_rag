import os
import sys
from pathlib import Path
from typing import List

from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory 
from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredMarkdownLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import (CHUNK_SIZE, CHUNK_OVERLAP, DOCS_DIR , CHROMA_DIR, COLLECTION_NAME,
                     RETRIEVER_FETCH_K, RETRIEVER_K, 
                    LLM_MODEL, EMBEDDING_MODEL, TEMPERATURE)


def load_Documents(docs_dir:str)-> list:
    path = Path(docs_dir)
    if not path.exists():
        raise FileNotFoundError(
            f" docs directory '{docs_dir} not found"
            "create it and crop your pdfs/.md files inside"
        )
    print(f" loading documents from: {path.resolve()}")

    # pdf loader
    pdf_loader = DirectoryLoader(
        docs_dir,
        glob = "**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True,
    )

    md_loader = DirectoryLoader(
        docs_dir,
        glob="**/*.md",
        loader_cls= UnstructuredMarkdownLoader,
        show_progress=True,
    )

    pdf_docs = pdf_loader.load()
    md_docs = md_loader.load()
    all_docs = pdf_docs + md_docs

    print(f" loaded {len(pdf_docs)} pdf pages and {len(md_docs)} markdown files"
          f"-> {len(all_docs)} total documents")
    
    return all_docs


def split_documents(documents:list)-> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function= len,
        add_start_index = True,
    )

    chunks = splitter.split_documents(documents)
    print(f"split into {len(chunks)} chunks"
          f"(size={CHUNK_SIZE}, overlap= {CHUNK_OVERLAP})")
    return chunks

          
def get_vectorstore(chunks: list, force_rebuild: bool = False):
    embeddings = OpenAIEmbeddings(model = EMBEDDING_MODEL)
    chroma_path = Path(CHROMA_DIR)

    if force_rebuild and chroma_path.exists():
        import shutil
        shutil.rmtree(chroma_path)
        print(" removed existing chromadb - rebuilding from scratch")

    if chroma_path.exists() and any(chroma_path.iterdir()) and not force_rebuild():
        print(f" loading existing chromadb from : {chroma_path.resolve()}")
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
        )
    else:
        print(F" building chromadb index(this may take a minute)....")
        vectorstore= Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_DIR,
            collection_metadata={"hnsw:space": "cosine"},
        )
        print(f" chromadb built and persisted to: {chroma_path.resolve()}")

    count = vectorstore._collection.count()
    print(f" collection '{COLLECTION_NAME}' contains {count} vectors")
    return vectorstore


def built_retriever(vectorstore):
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs = {
            "k": RETRIEVER_K,
            "fetch_k": RETRIEVER_FETCH_K,
        },
    )

SYSTEM_TEMPLATE= """You are a precise knowledge assistant for a private document collection.
Your ONLY source of truth is the context provoded below.

Rules you MUST follow:
1.Answer questions using ONLY information found in the context.
2. If the answer is not in the context, respond with exactly:
"I don't know - this information is not in the provided documents.
3. Never speculate, infer or use knowledge outside the provide context.
4. If the context partially answers the question, share what you found and clearly state what is missing.
5. Cite the source document name when possible (use the metadata).

Context:
{context}
"""

def build_qa_chain(retriever):
    """
    Build a ConversationalRetrievalChain with:
    - Custom anti-hallucination system prompt
    - Conversation memory (preserves chat history across turns)
    """
    llm = ChatOpenAI(
        model = LLM_MODEL,
        temperature= TEMPERATURE,
        streaming=True,
    )

    
    system_message_prompt = SystemMessagePromptTemplate(
        prompt = PromptTemplate(
            input_variables=["context"],
            template=SYSTEM_TEMPLATE,
        )
    )
    human_message_prompt = HumanMessagePromptTemplate(
        prompt = PromptTemplate(
            input_variables=["question"],
            template= "{question}",
        )
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        system_message_prompt, 
        human_message_prompt,
    ])

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory= memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        return_source_documents=True,
        verbose=False,
    )
    return chain

def print_sources(source_docs:list)-> None:
    """Pretty-print the retrieved source chunks."""
    seen = set()
    print("\n Sources:")
    for doc in source_docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page","")
        key = f"{source}:{page}"
        if key not in seen:
            seen.add(key)
            page_Str =f"(page {page})" if page != "" else ""
            print(f" {source}{page_Str}")

def run_Repl(chain)-> None:
    """Run an interactive Q&A loop with the knowledge oracle."""
    print("\n" + "="* 60)
    print(" Local knowledge oracle = type 'exit' to quit")
    print("="* 60 + "\n")

    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt,EOFError):
            print("\nBye!")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print("Bye!!")
            break

        print("\n Oracle: ", end="", flush=True)

        result = chain.invoke({"question": query})

        answer = result.get("answer", "")
        source_docs = result.get("source_documents", [])

        print()
        print_sources(source_docs)
        print()

