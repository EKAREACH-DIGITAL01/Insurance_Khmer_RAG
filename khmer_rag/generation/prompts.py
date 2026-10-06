"""Source context formatting and generator instructions."""

import time

from khmer_rag.language import REDIRECT_TEXT
from khmer_rag.text import clean_context_text_for_prompt


def build_context(results):
    context_start = time.time()
    context_parts = []
    for rank, (score, chunk) in enumerate(results, start=1):
        metadata = chunk.get("metadata", {})
        doc_name = metadata.get("document_name", "N/A")
        doc_type = metadata.get("document_type_kh", metadata.get("document_type", "N/A"))
        chapter = metadata.get("chapter_title", metadata.get("chapter", "N/A"))
        section = metadata.get("section_title", metadata.get("section", "N/A"))
        legal_unit = metadata.get("legal_unit", "N/A")
        release_date = metadata.get("release_date", "N/A")
        header_parts = [f"[Source {rank}] {doc_name}", f"ប្រភេទ: {doc_type}"]
        if release_date and release_date != "N/A":
            header_parts.append(f"កាលបរិច្ឆេទ: {release_date}")
        if chapter and chapter != "N/A":
            header_parts.append(f"ជំពូក/ផ្នែក: {chapter}")
        if section and section != "N/A":
            header_parts.append(f"ចំណុច: {section}")
        if legal_unit and legal_unit != "N/A":
            header_parts.append(f"ឯកតាច្បាប់: {legal_unit}")
        context_text = clean_context_text_for_prompt(chunk.get("text") or "")
        if not context_text:
            continue
        header = " | ".join(header_parts)
        context_parts.append(f"{header}\n{context_text}")
    context = "\n\n".join(context_parts)
    context_length = len(context)
    context_time = time.time() - context_start
    return context, context_length, context_time


SAILOR_STYLE_RULES = """
- Do NOT add a title or heading, do NOT use bold or any markdown formatting, and do NOT cite the source document's name, article/provision number, or any date (for example, do NOT write "(មាត្រា ១)", "(2023-10-17)", or "(តាម គោលការណ៍ណែនាំ...)"). Answer in plain Khmer prose, or a plain numbered/dashed list, exactly as the source presents it.
- Do NOT write any closing word or tag such as "(ចប់)", "(បញ្ចប់)", or a trailing "៕". End only with ។ on the final sentence.
- Do NOT add framing, emphasis, or any information that is not in the context. Match the wording and scope of the source -- do not expand, summarize away, or restructure its meaning."""


def _is_sailor(model_id):
    return "sailor" in (model_id or "").lower()


def build_prompt(question, context, low_confidence=False, model_id=""):
    relevance_check = ""
    if low_confidence:
        # Triggered when best_score sits below MIN_SCORE but above
        # OUT_OF_DOMAIN_THRESHOLD -- i.e. retrieval wasn't confident enough
        # to reject outright, but isn't a clean match either. The risk in
        # this band specifically isn't "no context was found" (the model
        # already handles that case via the rule below) -- it's that the
        # retrieved chunks share VOCABULARY with the question without
        # sharing its actual TOPIC (e.g. a meta-question about the
        # assistant itself landing on legal complaint-procedure documents
        # purely because both use the word "answer/respond"). A 7B model
        # under low temperature tends to pattern-match surface phrasing
        # from such chunks into a confident-sounding but wrong answer
        # rather than recognizing the mismatch, so this band gets an
        # explicit instruction to check topical alignment before relying
        # on the context at all.
        relevance_check = """
IMPORTANT: The retrieved context below has LOW relevance confidence for this
question. Before answering, first check whether the context is actually
ABOUT the same topic as the question -- not just whether it shares similar
words or phrasing. If the context discusses a different subject, or the
question is not really about Cambodian insurance law at all (for example,
a question about the assistant itself, or unrelated small talk), you MUST
reply with the "not found" message below instead of extracting a loosely
related answer.
"""

    not_found_message = f"រកមិនឃើញព័ត៌មាននេះនៅក្នុងឯកសារដែលបានផ្តល់។ {REDIRECT_TEXT}"

    # Sailor-only style correction (empty for every other model, so their
    # prompt is unchanged). Appended at the end of the rules list below.
    style_rules = SAILOR_STYLE_RULES if _is_sailor(model_id) else ""

    return f"""
You are a Khmer legal assistant.

Answer ONLY from the provided context.
{relevance_check}
Rules:

- Answer in Khmer.
- Do not invent facts.
- Preserve dates, fees, deadlines and requirements exactly.
- Use the source wording as much as possible. Prefer copying the relevant Khmer phrase/sentence from the context instead of paraphrasing it.
- If only one source directly answers the question, use only that source. Do not combine other sources just because they were retrieved.
- Provide a concise professional answer. If the context lists multiple directly relevant conditions, sub-points, exceptions, or requirements, enumerate them once and keep their order.
- Do not repeat the same sentence, phrase, list item, or explanation.
- Do not write a "Sources", "ប្រភព", "មើលប្រភពឯកសារ", citation, disclaimer, HTML, button text, or UI/action text. The application will show sources separately after your answer.
- Always finish your answer with a complete sentence that ends in the Khmer sentence-final sign ។ (or ៕). Even when the answer is a list, the final item must end with ។. Never stop on a word, bracket, or list item without it.
- CRITICAL: If the question names a specific provision (មាត្រា N for a law or sub-decree, ប្រការ N for a prakas), you MUST reproduce that provision's ENTIRE text from the context word-for-word -- its heading line AND every following sentence, clause, condition, and sub-point. This rule applies EVEN IF the question is phrased as "និយាយអំពីអ្វី" / "មានន័យដូចម្តេច" / "about what" -- such phrasing does NOT permit a one-line topic summary. A reply that gives only the provision's subject or title is WRONG and forbidden. Quote the full provision.
- Only combine information from multiple sources if EACH of them directly addresses the question asked. Do not stitch together unrelated facts from different sources just because they were retrieved -- if most of the retrieved sources don't actually answer the question, say so instead of forcing a combined answer out of all of them.
- Do not reinterpret a number, rate, or statistic for a purpose other than what it was stated for in the source (for example, a mortality or incidence rate expressed "per 1,000 people" is NOT the same as a total population count -- never restate a rate's denominator as if it were the literal answer to a "how many" question).
- Do not recommend, endorse, or compare any specific insurance product, company, or policy, even if the context mentions one and the question asks for a recommendation.
- Do not calculate, estimate, or infer a premium, payout, deductible, or any monetary amount that is not stated explicitly and directly in the context for that exact purpose.
- Do not give personalized financial or legal advice (e.g. "should I buy X", "what should I do in my situation"). If the question asks for a personal recommendation, a calculation, or advice tailored to the user's situation, decline and reply with the message below instead of attempting an answer.{style_rules}
- If the answer is not found in the context, OR the question asks for a product recommendation, a premium/payout calculation, or personalized advice, reply EXACTLY with this message and nothing else:

{not_found_message}

Context:
{context}

Question:
{question}

Answer:
"""
