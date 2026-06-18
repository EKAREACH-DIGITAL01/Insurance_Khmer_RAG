import json
import time
import faiss
import requests
import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer

LLAMA_SERVER_URL = "http://localhost:8080/v1"


# Get the model that user input
def get_loaded_model():
    try:
        response = requests.get(
            f"{LLAMA_SERVER_URL}/models",
            timeout=5,
        )

        response.raise_for_status()

        data = response.json()

        return data["data"][0]["id"]

    except Exception:
        return "No Model Loaded"


st.set_page_config(
    page_title="Khmer Legal Chatbot",
    layout="wide",
)

MODEL_NAME = get_loaded_model()

if MODEL_NAME == "No Model Loaded":
    st.error("llama-server is not running.")
    st.stop()

left_col, right_col = st.columns([2, 1])

with left_col:
    st.title("Khmer Legal Chatbot")
    st.caption(f"RAG + {MODEL_NAME}")


client = OpenAI(
    base_url=LLAMA_SERVER_URL,
    api_key="not-needed",
)

TOP_K = 3
MIN_SCORE = 0.50

FAISS_PATH = "vector_db/khmer_insurance.index"
METADATA_PATH = "vector_db/khmer_insurance_metadata.json"

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

if "has_result" not in st.session_state:
    st.session_state["has_result"] = False


# RETRIEVAL
def retrieve(query):
    embed_start = time.time()

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True,
    )

    embed_time = time.time() - embed_start

    search_start = time.time()

    scores, indices = index.search(
        query_embedding,
        k=TOP_K,
    )

    search_time = time.time() - search_start

    results = []

    for score, idx in zip(scores[0], indices[0]):
        results.append(
            (
                score,
                chunks[idx],
            )
        )

    return (
        results,
        embed_time,
        search_time,
    )


# BUILD CONTEXT
def build_context(results):
    context_start = time.time()

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

    context = "\n\n".join(context_parts)

    context_length = len(context)

    context_time = time.time() - context_start

    return (
        context,
        context_time,
        context_length,
    )


def stream_answer(
    question,
    context,
    diagnostics,
):
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

    diagnostics["prompt_length"] = len(prompt)

    start = time.time()
    first_token = True

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.2,
        max_tokens=512,
        stream=True,
    )

    total_chars = 0

    for chunk in response:

        delta = chunk.choices[0].delta.content

        if delta:

            total_chars += len(delta)

            if first_token:
                diagnostics["first_token_time"] = time.time() - start
                first_token = False

            yield delta

    diagnostics["generated_chars"] = total_chars

    diagnostics["generation_time"] = time.time() - start


# UI
with left_col:
    question = st.text_area("Ask a Khmer legal question:")

    if st.button("Ask") and question.strip():
        status = st.status(
            "Processing...",
            expanded=True,
        )
        total_start = time.time()

        with st.spinner("Searching..."):
            retrieval_start = time.time()

            status.write("Searching vector database...")
            (
                results,
                embed_time,
                search_time,
            ) = retrieve(question)

            if not results:
                st.error("រកមិនឃើញព័ត៌មាននេះនៅក្នុងឯកសារដែលបានផ្តល់។")
                st.stop()

            best_score = results[0][0]

            if best_score < MIN_SCORE:
                st.warning(f"Low confidence retrieval (best score = {best_score:.4f})")

            retrieval_time = time.time() - retrieval_start

            status.write("Building context...")
            (
                context,
                context_time,
                context_length,
            ) = build_context(results)

            generation_start = time.time()

            status.write("Generating answer...")
            st.subheader("Answer")

            diagnostics = {}

            st.write_stream(
                stream_answer(
                    question,
                    context,
                    diagnostics,
                )
            )

            status.update(
                label="Completed",
                state="complete",
            )

            generation_time = time.time() - generation_start
            total_time = time.time() - total_start
            st.session_state["diagnostics"] = {
                "embed_time": embed_time,
                "search_time": search_time,
                "retrieval_time": retrieval_time,
                "best_score": float(best_score),
                "context_length": context_length,
                "context_time": context_time,
                "generation_time": generation_time,
                "total_time": total_time,
                "results": results,
                "diagnostics": diagnostics,
            }

            st.session_state["has_result"] = True

        st.markdown("### Sources Used")
        for rank, (score, chunk) in enumerate(results, start=1):
            metadata = chunk.get("metadata", {})

            st.markdown(f"""
        {rank}. **{metadata.get('document_name', 'Unknown Document')}**
        - Score: {score:.4f}
        - Chunk: {chunk['chunk_id']}
        """)

        st.subheader("Source Details")

        for rank, (score, chunk) in enumerate(
            results,
            start=1,
        ):
            with st.expander(f"Rank {rank} | Score {score:.4f}"):

                metadata = chunk.get("metadata", {})

                st.write(
                    {
                        "similarity_score": round(
                            float(score),
                            4,
                        ),
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

with right_col:
    diag = st.session_state.get(
        "diagnostics",
        {},
    )
    st.header("Diagnostics")

    st.subheader("System")

    st.write(
        {
            "model": MODEL_NAME,
            "embedding_model": EMBEDDING_MODEL_PATH,
            "embedding_device": str(embedding_model.device),
            "top_k": TOP_K,
        }
    )

    if st.session_state.get("has_result", False):

        st.subheader("Retrieval")

        st.write(
            {
                "embedding_time_sec": round(
                    diag["embed_time"],
                    3,
                ),
                "faiss_time_sec": round(
                    diag["search_time"],
                    3,
                ),
                "retrieval_time_sec": round(
                    diag["retrieval_time"],
                    3,
                ),
                "best_score": round(
                    float(diag["best_score"]),
                    4,
                ),
            }
        )

        st.subheader("Context")

        st.write(
            {
                "context_length_chars": diag["context_length"],
                "context_build_time_sec": round(
                    diag["context_time"],
                    3,
                ),
                "prompt_length_chars": diag["diagnostics"].get(
                    "prompt_length",
                    0,
                ),
            }
        )

        st.subheader("Generation")

        st.write(
            {
                "first_token_time_sec": round(
                    diag["diagnostics"].get(
                        "first_token_time",
                        0,
                    ),
                    3,
                ),
                "generation_time_sec": round(
                    diag["diagnostics"].get(
                        "generation_time",
                        0,
                    ),
                    3,
                ),
                "total_time_sec": round(
                    diag["total_time"],
                    3,
                ),
                "generated_chars": diag["diagnostics"].get(
                    "generated_chars",
                    0,
                ),
            }
        )

        st.subheader("Chunks")

        for rank, (score, chunk) in enumerate(
            diag["results"],
            start=1,
        ):
            st.write(
                {
                    "rank": rank,
                    "score": round(
                        float(score),
                        4,
                    ),
                    "chunk_length": len(chunk["text"]),
                    "chunk_id": chunk["chunk_id"],
                }
            )
