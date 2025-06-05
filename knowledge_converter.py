import crawl4ai
import streamlit as st

#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document

GEMINI_API_KEY = st.secrets["CREDENTIAL"]["GEMINI_API_KEY"]


def convert_to_vector(items, urls):

    documents = []

    print(f'total documents: {len(items)}')
    for idx, item in enumerate(items):

        print(item)

        doc = Document(page_content=item, metadata={"web source": urls[idx]})
        documents.append(doc)

    print(f'total documents: {len(documents)}')
    

    
    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY) # define the embedding

    #shutil.rmtree(PERSIST_PATH, ignore_errors=True)
    #try:
    #    vector_store = Chroma(embeddings=embeddings, persist_directory=PERSIST_PATH)
    #    vector_store.delete()
    #    print('inner deleted')
    #except:
    #    pass

    #db = Chroma.from_documents(
    #    docs, embeddings, persist_directory=PERSIST_PATH)
    db = FAISS.from_documents(docs, embeddings)
    
    print('Converted to Knowledge!!')

    return db