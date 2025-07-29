import os
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()
API_KEY = os.getenv("PINECONE_API_KEY")
REGION = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-index")

# 1) Pinecone client (v3)
pc = Pinecone(api_key=API_KEY)

# 2) Ensure index exists (dim must match your embedding model)
DIM = 768  # using all-mpnet-base-v2 (outputs 768-dim vectors)
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=REGION),
    )

# 3) Load and split your documents (PDF path can be changed)
loader = PyPDFLoader("data/basic_medical_knowledge.pdf")
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
docs = splitter.split_documents(pages)

# 4) Embeddings — 768-dim to match index
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# 5) Upsert to Pinecone using the LangChain adapter for v3
PineconeVectorStore.from_documents(
    documents=docs,
    embedding=embeddings,
    index_name=INDEX_NAME,
)

print(f"✅ Indexed {len(docs)} chunks into '{INDEX_NAME}'.")

