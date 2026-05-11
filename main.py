from config import DOCS_DIR
import os, sys
from dotenv import load_dotenv
from oracle import (load_Documents, get_vectorstore, split_documents,
                    built_retriever, build_qa_chain, run_Repl)

def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("X OPEN_API_KEY environment variable is not set.")
        print(" Run: export OPENAI_API_KEY= 'sk-...'")
        sys.exit(1)

    force_rebuild = "--rebuild" in sys.argv

    #pipeline
    documents = load_Documents(DOCS_DIR)
    chunks = split_documents(documents)
    vectorstore = get_vectorstore(chunks, force_rebuild=force_rebuild)
    retriever = built_retriever(vectorstore)
    chain = build_qa_chain(retriever)

    run_Repl(chain)


if __name__ == "__main__":
    main()



