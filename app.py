import json
import time
import faiss
import numpy as np
import streamlit as st
import ollama
from sentence_transformers import SentenceTransformer

OLLAMA_MODEL = "qwen3.5:4b"
TOP_K = 3
MAX_OUTPUT_TOKENS = 1024

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="Khmer Legal Chatbot",
    layout="wide"
)

# -----------------------------
# LOAD ENV
# -----------------------------



# -----------------------------
# CACHED RESOURCES
# -----------------------------

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "BAAI/bge-m3"
    )

@st.cache_resource
def load_faiss_index():

    return faiss.read_index(
        "vector_db/khmer_insurance.index"
    )

@st.cache_resource
def load_metadata():

    with open(
        "vector_db/khmer_insurance_metadata.json",
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)

# Load resources
embedding_model = load_embedding_model()

index = load_faiss_index()

chunks = load_metadata()

# -----------------------------
# UI
# -----------------------------

st.title("Khmer Legal Chatbot")

st.write(
    "RAG chatbot test using Khmer legal documents."
)

query = st.text_area(
    "Ask a question in Khmer:"
)

# -----------------------------
# ASK BUTTON
# -----------------------------
if st.button("Ask") and query.strip():

    with st.spinner("Searching documents..."):

        total_start = time.time()

        # -----------------------------
        # EMBED QUERY
        # -----------------------------

        start = time.time()

        query_embedding = embedding_model.encode(
            [query],
            normalize_embeddings=True
        )

        query_embedding = np.array(
            query_embedding
        ).astype("float32")

        st.write(
            f"Embedding time: {time.time() - start:.2f}s"
        )

        # -----------------------------
        # VECTOR SEARCH
        # -----------------------------

        start = time.time()

        distances, indices = index.search(
            query_embedding,
            k=TOP_K
        )

        st.write(
            f"FAISS search time: {time.time() - start:.2f}s"
        )

        # -----------------------------
        # BUILD CONTEXT
        # -----------------------------

        retrieved_chunks = []

        for rank, i in enumerate(indices[0]):

            chunk = chunks[i]

            retrieved_chunks.append(
                f"""
[Source {rank + 1}]
Document ID: {chunk["document_id"]}
Page: {chunk["page"]}
Chunk ID: {chunk["chunk_id"]}

{chunk["text"]}
"""
            )

        context = "\n\n".join(
            retrieved_chunks
        )

        st.write(
            f"Prompt context length: {len(context)} characters"
        )

        # -----------------------------
        # RAG PROMPT
        # -----------------------------

        prompt = f"""
You are a Khmer legal document assistant.

Answer the user's question using ONLY the provided context.

Rules:
- Answer directly.
- Do not show reasoning.
- Do not think step by step.
- Do NOT use outside knowledge.
- Do NOT invent information.
- Do NOT hallucinate laws.
- If the answer is not found in the context, say:
  "រកមិនឃើញព័ត៌មាននេះនៅក្នុងឯកសារដែលបានផ្តល់។"
- Answer in Khmer.
- Keep the answer short and clear.
- Mention page numbers if relevant.

Context:
{context}

Question:
{query}
"""

        # -----------------------------
        # LOCAL QWEN GENERATION
        # -----------------------------

        response = None

        try:

            start = time.time()

            response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Khmer legal assistant. "
                        "Answer directly and briefly in Khmer. "
                        "Do not show thinking. "
                        "Do not explain your reasoning."
                    )
                },
                {
                    "role": "user",
                    "content": prompt + "\n\n/no_think"
                }
            ],
            options={
                "temperature": 0.2,
                "num_predict": 512,
                "num_ctx": 2048
            }
        )

            st.write(
                f"LLM generation time: {time.time() - start:.2f}s"
            )

        except Exception as e:

            st.error("OLLAMA ERROR:")
            st.code(str(e))

            st.stop()

        if response is None:

            st.error(
                "Ollama request failed."
            )

            st.stop()

        st.write(
            f"Total time: {time.time() - total_start:.2f}s"
        )

    st.subheader("Answer")

    message = response["message"]

    answer = message.get(
        "content",
        ""
    ).strip()

    thinking = message.get(
        "thinking",
        ""
    ).strip()

    if not answer and thinking:
        answer = (
            "រកមិនឃើញចម្លើយចុងក្រោយពីម៉ូដែល។ "
            "ម៉ូដែលបានបង្កើតតែ thinking output ប៉ុណ្ណោះ។"
        )

    st.subheader("Retrieved Sources")

    for rank, i in enumerate(indices[0]):

        chunk = chunks[i]

        with st.expander(
            f"Rank {rank + 1} | "
            f"Page {chunk['page']} | "
            f"Distance "
            f"{distances[0][rank]:.4f}"
        ):

            st.write(chunk["text"])