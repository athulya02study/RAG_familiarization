import os
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
client = OpenAI(api_key=os.environ["GROQ_API_KEY"],
                base_url="https://api.groq.com/openai/v1")

# 1. Documents (our "knowledge base"), already chunked
chunks = [
    "Athulya's company offers 20 days of paid leave per year.",
    "The office is open Monday to Friday, 9 AM to 5 PM.",
    "Employees can work from home on Wednesdays and Fridays.",
    "Expense claims must be submitted within 30 days with receipts.",
    "The IT helpdesk can be reached at extension 4455.",
]

# 2. Index: turn chunks into vectors
vectorizer = TfidfVectorizer()
chunk_vectors = vectorizer.fit_transform(chunks)

# 3. Retrieve: find the most similar chunks to the question
def retrieve(question, k=2):
    q_vec = vectorizer.transform([question])
    scores = cosine_similarity(q_vec, chunk_vectors)[0]
    top = scores.argsort()[::-1][:k]
    return [chunks[i] for i in top]

# 4. Generate: answer using only the retrieved context
def ask(question):
    context = "\n".join(retrieve(question))
    prompt = (f"Answer using ONLY the context below. "
              f"If the answer isn't there, say you don't know.\n\n"
              f"Context:\n{context}\n\nQuestion: {question}")
    resp = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content

while True:
    q = input("\nAsk (or 'quit'): ")
    if q.lower() == "quit":
        break
    print("Answer:", ask(q))