import os
from dotenv import load_dotenv
from openai import OpenAI

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory="db",
    embedding_function=embedding
)

print("RAG Ready!")

while True:
    question = input("\nAsk: ")

    if question.lower() == "exit":
        break

    docs = db.similarity_search(question, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "Answer only from the provided context."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:{question}"
            }
        ],
        temperature=0.2
    )

    print("\nAnswer:\n")
    print(response.choices[0].message.content)