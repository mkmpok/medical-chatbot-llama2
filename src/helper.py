from langchain.embeddings import HuggingFaceEmbeddings

def download_hugging_face_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


