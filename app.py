import json
import time
import faiss
import numpy as np
import streamlit as st
import ollama
from sentence_transformers import SentenceTransformer

OLLAMA_MODEL = "yxchia/seallms-v3-7b:Q4_K_M"
TOP_K = 5
MIN_SCORE = 0.50

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

        valid_results = []

        for score, idx in zip(
            distances[0],
            indices[0]
        ):

            if score >= MIN_SCORE:

                valid_results.append(
                    (score, idx)
                )

        if not valid_results:

            st.subheader("Answer")

            st.write(
                "រកមិនឃើញព័ត៌មាននេះនៅក្នុងឯកសារដែលបានផ្តល់។"
            )

            st.stop()

        st.write(
            f"FAISS search time: {time.time() - start:.2f}s"
        )

        # -----------------------------
        # BUILD CONTEXT
        # -----------------------------

        retrieved_chunks = []

        sources = []

        for rank, (score, i) in enumerate(
            valid_results
        ):

            chunk = chunks[i]

            sources.append(
                (
                    chunk["chunk_id"],
                    score
                )
            )

            retrieved_chunks.append(
                (
                    f"[Source {rank + 1}]\n"
                    f"Document ID: {chunk['document_id']}\n"
                    f"Chunk ID: {chunk['chunk_id']}\n\n"
                    f"{chunk['text']}"
                )
            )

        context = "\n\n".join(
            retrieved_chunks
        )

        st.subheader("Retrieved Chunks")

        for rank, (score, i) in enumerate(
            valid_results
        ):

            chunk = chunks[i]

            st.markdown(
                        f"""
                **Rank {rank + 1}**
                - Score: {score:.4f}
                - Document: {chunk['document_id']}
                - Category: {chunk['category']}
                - Chunk ID: {chunk['chunk_id']}
                """
            )

        st.write(
            f"Prompt context length: {len(context)} characters"
        )

        # -----------------------------
        # RAG PROMPT
        # -----------------------------

        prompt = f"""
        You are a Khmer legal assistant.

        Answer ONLY using the provided context.

        Rules:

        - Answer in Khmer.
        - Do not invent facts.
        - If multiple chunks contain relevant information, combine them.
        - If dates, fees, deadlines, authorities, procedures or requirements are mentioned, preserve them exactly.
        - Do not add outside legal knowledge.
        - If the answer is not explicitly supported by the context, reply exactly:

        រកមិនឃើញព័ត៌មាននេះនៅក្នុងឯកសារដែលបានផ្តល់។

        Context:
        {context}

        Question:
        {query}

        Answer:
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
                        "content": prompt
                    }
                ],
                options={
                    "temperature": 0.2,
                    "num_predict": 512,
                    "num_ctx": 4096
                }
            )

            st.subheader("RAW RESPONSE")
            st.json(response)

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

    answer = (
        message.get("content")
        or ""
    ).strip()

    thinking = (
        message.get("thinking")
        or ""
    ).strip()

    if not answer and thinking:
        answer = thinking

    if not answer:
        answer = "No response generated."

    st.write(answer)

    st.markdown(
        "**Sources:**\n\n"
        + "\n".join(
            f"- {chunk_id} ({score:.4f})"
            for chunk_id, score in sources
        )
    )

    for rank, (score, i) in enumerate(
        valid_results
    ):

        chunk = chunks[i]

        with st.expander(
            f"Rank {rank + 1} | Score {score:.4f}"
        ):

            st.caption(
                f"Document: {chunk['document_id']}"
            )

            st.caption(
                f"Category: {chunk['category']}"
            )

            st.caption(
                f"Chunk: {chunk['chunk_id']}"
            )

            st.write(
                chunk["text"]
            )