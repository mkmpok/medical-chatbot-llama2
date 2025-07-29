import os
from dotenv import load_dotenv
from flask import Flask, render_template, request

from pinecone import Pinecone
from langchain_community.llms import CTransformers
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# ------------------ Load environment variables ------------------
load_dotenv()
API_KEY = os.getenv("PINECONE_API_KEY")
REGION = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-index")

# ------------------ Setup Pinecone ------------------
pc = Pinecone(api_key=API_KEY)

# ------------------ Setup Embeddings ------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# ------------------ Connect to Vector Store ------------------
vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

# ------------------ Setup Prompt Template ------------------
prompt_template = """
You are a helpful and caring medical assistant. Answer the user's current health-related question using the following context and previous chat history.

Only provide the assistant's response. Do not repeat the user's words. Do not show labels like "User:" or "Assistant:".

[Context]
{context}

[Chat History]
{chat_history}

[User Question]
{question}

[Assistant Response]
"""




PROMPT = PromptTemplate.from_template(prompt_template)

# ------------------ Load Local LLaMA2 Model ------------------
llm = CTransformers(
    model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",  # Make sure this path is correct
    model_type="llama",
    config={
        "max_new_tokens": 100,
        "temperature": 0.2,
        "top_p": 0.9
    }
)

# ------------------ Conversational Chain with Memory ------------------
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": PROMPT},
    return_source_documents=False,
    output_key="answer"  # 🔥 Ensures no "multiple output keys" error
)

# ------------------ Flask Web App ------------------
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    user_input = request.form["msg"]
    print(f"User: {user_input}")
    try:
        result = qa_chain({"question": user_input})
        response = result["answer"]

        # ✅ Extract only the assistant's actual reply if "Assistant:" is included
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()

        print("Bot:", response)
        return str(response)
    except Exception as e:
        print("Error:", e)
        return "Sorry, something went wrong."


# ------------------ Run Server ------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

