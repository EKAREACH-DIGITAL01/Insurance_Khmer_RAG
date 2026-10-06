"""Verbatim legal provision lookup in retrieved context."""

import re

from khmer_rag.language import (
    ARTICLE_REF_PATTERN,
    REDIRECT_TEXT,
    to_khmer_digits,
)
from khmer_rag.text import clean_provision_text


def resolve_provision_lookup(question, context_results):
    """Deterministic answer for questions that explicitly name a provision
    (មាត្រា N / ប្រការ N). Bypasses the local model, which both summarizes such
    provisions down to their topic AND hallucinates scope not in the text
    (e.g. inventing "សេវា ឬការអះអាង..." onto មាត្រា ១០), so the provision's own
    words are the most faithful, complete answer to "what does it say".

    Returns:
      - a list of verbatim provision texts, when the chunk(s) for the named
        provision(s) -- from the document the question names -- were retrieved;
      - a one-element list with a "no such provision" message, when the
        question names a document but that document has no such provision in
        context (e.g. asking for មាត្រា ១០០ of a 67-article law). This is what
        stops a same-numbered provision from ANOTHER document being returned;
      - None, when no provision was named -- fall back to the model.
    """
    refs = []
    seen = set()
    for keyword, num in ARTICLE_REF_PATTERN.findall(question):
        # Corpus headings use Khmer digits; normalize so a typed Arabic number
        # ("មាត្រា 67") matches "មាត្រា ៦៧".
        num = to_khmer_digits(num)
        if (keyword, num) not in seen:
            seen.add((keyword, num))
            refs.append((keyword, num))
    if not refs:
        return None

    qn = re.sub(r"\s+", "", question)
    # Document titles the question explicitly names, matched against the known
    # document_name of every retrieved chunk. When the question names a
    # document, a provision is only accepted if it comes from that document.
    named_docs = {
        chunk.get("metadata", {}).get("document_name", "")
        for _score, chunk in context_results
        if re.sub(r"\s+", "", chunk.get("metadata", {}).get("document_name", ""))
        and re.sub(r"\s+", "", chunk.get("metadata", {}).get("document_name", "")) in qn
    }

    texts = []
    for keyword, num in refs:
        # Corpus heading: "មាត្រា ១០ - ...". Match keyword + number as a whole
        # unit; the lookahead stops "១០" from also matching "១០០"/"១០២".
        heading = re.compile(
            r"^\s*" + re.escape(keyword) + r"\s*" + re.escape(num) + r"(?![០១២៣៤៥៦៧៨៩0-9])"
        )
        match = None
        for _score, chunk in context_results:
            text = (chunk.get("text") or "").strip()
            if not heading.match(text):
                continue
            doc = chunk.get("metadata", {}).get("document_name", "")
            if named_docs and doc not in named_docs:
                continue  # right number, wrong document
            match = clean_provision_text(text)
            break
        if match is not None:
            texts.append(match)
        elif named_docs:
            # Question named a document, but it has no such provision: don't
            # fall through to a same-numbered provision from another law.
            doc_title = next(iter(named_docs))
            return [f"រកមិនឃើញ{keyword} {num} នៅក្នុង{doc_title}ឡើយ។ {REDIRECT_TEXT}"]
    return texts or None
