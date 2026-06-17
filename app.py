import json
import time
import faiss
from openai import OpenAI
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

LLAMA_SERVER_URL = "http://localhost:8080/v1"

client = OpenAI(
    base_url=LLAMA_SERVER_URL,
    api_key="not-needed",
)


TOP_K = 5
MIN_SCORE = 0.50

FAISS_PATH = "vector_db/khmer_insurance.index"
METADATA_PATH = "vector_db/khmer_insurance_metadata.json"

# STREAMLIT
st.set_page_config(
    page_title="Khmer Legal Chatbot",
    layout="wide",
)

st.title("Khmer Legal Chatbot")
st.caption("RAG + SeaLLM-v3-7B GGUF")


EMBEDDING_MODEL_PATH = "models/retriever/BAAI-BGE-M3"


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_PATH)


# LOAD FAISS
@st.cache_resource
def load_index():
    return faiss.read_index(FAISS_PATH)


@st.cache_resource
def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


embedding_model = load_embedding_model()
index = load_index()
chunks = load_metadata()


# RETRIEVAL
def retrieve(query):
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32",
    )

    scores, indices = index.search(
        query_embedding,
        k=TOP_K,
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):
        results.append(
            (
                score,
                chunks[idx],
            )
        )

    return results


# BUILD CONTEXT
def build_context(results):

    context_parts = []

    for rank, (score, chunk) in enumerate(results, start=1):

        metadata = chunk.get("metadata", {})

        context_parts.append(f"""
[Source {rank}]

Document Name:
{metadata.get("document_name", "N/A")}

Document Type:
{metadata.get("document_type_kh", metadata.get("document_type", "N/A"))}

Release Date:
{metadata.get("release_date", "N/A")}

Chapter:
{metadata.get("chapter_title", metadata.get("chapter", "N/A"))}

Section:
{metadata.get("section_title", metadata.get("section", "N/A"))}

Legal Unit:
{metadata.get("legal_unit", "N/A")}

Chunk ID:
{chunk["chunk_id"]}

Content:
{chunk["text"]}
""")

    return "\n\n".join(context_parts)


def generate_answer(question, context):

    prompt = f"""
You are a Khmer legal assistant.

Answer ONLY from the provided context.

Rules:

- Answer in Khmer.
- Do not invent facts.
- Preserve dates, fees, deadlines and requirements exactly.
- If answer not found, reply:

រកមិនឃើញព័ត៌មាននេះនៅក្នុងឯកសារដែលបានផ្តល់។

Context:
{context}

Question:
{question}

Answer:
"""

    try:
        response = client.chat.completions.create(
            model="SeaLLM",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.2,
            max_tokens=512,
        )

    except Exception as e:
        return f"LLM Error: {str(e)}"

    return response.choices[0].message.content.strip()


# UI
question = st.text_area("Ask a Khmer legal question:")

if st.button("Ask") and question.strip():
    total_start = time.time()

    with st.spinner("Searching..."):
        retrieval_start = time.time()

        results = retrieve(question)

        if not results:
            st.error("រកមិនឃើញព័ត៌មាននេះនៅក្នុងឯកសារដែលបានផ្តល់។")
            st.stop()

        best_score = results[0][0]

        if best_score < MIN_SCORE:
            st.warning(f"Low confidence retrieval (best score = {best_score:.4f})")

        retrieval_time = time.time() - retrieval_start

        context = build_context(results)

        generation_start = time.time()

        answer = generate_answer(
            question,
            context,
        )

        generation_time = time.time() - generation_start

    st.subheader("Answer")
    st.write(answer)

    st.markdown("### Sources Used")

    for rank, (score, chunk) in enumerate(results, start=1):

        metadata = chunk.get("metadata", {})

        st.markdown(f"""
    {rank}. **{metadata.get('document_name', 'Unknown Document')}**
    - Score: {score:.4f}
    - Chunk: {chunk['chunk_id']}
    """)
    st.subheader("Performance")

    st.write(f"Retrieval Time: {retrieval_time:.2f}s")

    st.write(f"Generation Time: {generation_time:.2f}s")

    st.write(f"Total Time: {time.time() - total_start:.2f}s")

    st.subheader("Source Details")

    for rank, (score, chunk) in enumerate(
        results,
        start=1,
    ):
        with st.expander(f"Rank {rank} | Score {score:.4f}"):

            metadata = chunk.get("metadata", {})

            st.write(
                {
                    "similarity_score": round(score, 4),
                    "document_name": metadata.get("document_name"),
                    "document_type": metadata.get(
                        "document_type_kh",
                        metadata.get("document_type"),
                    ),
                    "release_date": metadata.get("release_date"),
                    "chapter": metadata.get("chapter"),
                    "chapter_title": metadata.get("chapter_title"),
                    "section": metadata.get("section"),
                    "section_title": metadata.get("section_title"),
                    "legal_unit": metadata.get("legal_unit"),
                    "chunk_id": chunk["chunk_id"],
                }
            )

            st.markdown("---")

            st.write(chunk["text"])
