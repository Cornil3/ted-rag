from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

DATA_PATH = "ted_talks_en.csv"
INDEX_NAME = "ted-talks"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "RPRTHPB-text-embedding-3-small"

def load_documents(path: str):
    loader = CSVLoader(file_path=path, encoding="utf-8")
    return loader.load()

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    return splitter.split_documents(documents)

def init_vectorstore():
    pc = Pinecone()
    index = pc.Index(INDEX_NAME)

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )
    return index, embeddings

def upload_vectors():
    documents = load_documents(DATA_PATH)
    chunks = split_documents(documents)
    index, embeddings = init_vectorstore()

    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME,
    )

if __name__ == "__main__":
    upload_vectors()
