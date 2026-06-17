import json
import time
import faiss
import numpy as np
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# CONFIG
MODEL_NAME = "SeaLLMs/SeaLLMs-v3-7B-Chat"

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
st.caption("RAG + SeaLLMs-v3-7B-Chat")


# LOAD EMBEDDING MODEL
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("BAAI/bge-m3")


# LOAD FAISS
@st.cache_resource
def load_index():
    return faiss.read_index(FAISS_PATH)


@st.cache_resource
def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# LOAD LLM
@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
    )

    return tokenizer, model


embedding_model = load_embedding_model()
index = load_index()
chunks = load_metadata()

tokenizer, model = load_llm()


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


# GENERATE ANSWER
def generate_answer(question, context):
    prompt = f"""

You are a Khmer legal assistant.

Answer ONLY from the provided context.

Rules:

* Answer in Khmer.
* Do not invent facts.
* Preserve dates, fees, deadlines and requirements exactly.
* If answer not found, reply:

រកមិនឃើញព័ត៌មាននេះនៅក្នុងឯកសារដែលបានផ្តល់។

Context:
{context}

Question:
{question}

Answer:
"""

    messages = [
        {
            "role": "system",
            "content": "You are a Khmer legal assistant.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.2,
        do_sample=True,
    )

    answer = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )

    return answer


# UI
question = st.text_area("Ask a Khmer legal question:")

if st.button("Ask") and question.strip():
    total_start = time.time()

    with st.spinner("Searching..."):
        retrieval_start = time.time()

        results = retrieve(question)

        best_score = results[0][0]

        if best_score < MIN_SCORE:
            st.warning(f"Low confidence retrieval (best score = {best_score:.4f})")

        retrieval_time = time.time() - retrieval_start

        if not results:
            st.error("រកមិនឃើញព័ត៌មាននេះនៅក្នុងឯកសារដែលបានផ្តល់។")
            st.stop()

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
