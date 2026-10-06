"""Source cleanup and answer normalization. No UI or model dependencies."""

import re

GENERATED_SOURCE_SECTION_PATTERN = re.compile(
    r"^\s*(?:មើលប្រភពឯកសារ|ប្រភពឯកសារ|ប្រភព|Sources?|ឯកសារយោង)(?:\s*[:：-]|$)",
    re.IGNORECASE,
)

HTML_ARTIFACT_PATTERN = re.compile(
    r"(?:data-tip=|aria-label=|<button|</button|<svg|</svg|<div|</div|<ul|</ul|<li|</li)",
    re.IGNORECASE,
)

CONTEXT_SOURCE_NOISE_PATTERN = re.compile(
    r"^\s*(?:"
    r"Source file:|OCR method:|\[Page \d+\]|"
    r"Source URL:|URL:|File URL:|Download(?: file)?:|"
    r"Logo:|Image:|Image file:|Asset:|https?://"
    r")",
    re.IGNORECASE,
)

CONTEXT_URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)


def _dedupe_consecutive_lines(lines):
    deduped = []
    recent = []
    for line in lines:
        key = re.sub(r"\s+", "", line)
        if key and key in recent[-3:]:
            continue
        deduped.append(line)
        if key:
            recent.append(key)
    return deduped


def _dedupe_repeated_sentences(text):
    """Remove short local-model loops without deleting legitimate lists."""
    pieces = re.split(r"([។៕!?])", text)
    sentences = []
    for i in range(0, len(pieces), 2):
        body = pieces[i]
        punct = pieces[i + 1] if i + 1 < len(pieces) else ""
        if body or punct:
            sentences.append(body + punct)

    out = []
    recent = []
    for sentence in sentences:
        key = re.sub(r"\s+", "", sentence)
        if len(key) > 30 and key in recent[-4:]:
            continue
        out.append(sentence)
        if key:
            recent.append(key)
    return "".join(out)


def _trim_to_last_complete_sentence(text):
    """Hide a dangling tail when generation stopped mid-sentence.

    This does not force a period onto incomplete text. It keeps the last
    complete Khmer sentence when doing so preserves most of the answer, which
    reads better and is less misleading during a live demo.
    """
    text = text.strip()
    if not text or text.endswith(("។", "៕", ".", "!", "?")):
        return text
    last = max(text.rfind("។"), text.rfind("៕"))
    if last >= 0 and last >= len(text) * 0.55:
        return text[: last + 1].strip()
    return text


def normalize_answer_text(text, final=True):
    """Clean model output before display/storage."""
    if not text:
        return ""
    cleaned_lines = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if HTML_ARTIFACT_PATTERN.search(line):
            break
        if GENERATED_SOURCE_SECTION_PATTERN.search(line):
            break
        cleaned_lines.append(line)
    cleaned_lines = _dedupe_consecutive_lines(cleaned_lines)
    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n\s*\n+", "\n", cleaned)
    cleaned = re.sub(r"(?m)^\s*[-•]\s*$\n?", "", cleaned)
    cleaned = _dedupe_repeated_sentences(cleaned)
    cleaned = fix_common_generation_typos(cleaned)
    if final:
        cleaned = _trim_to_last_complete_sentence(cleaned)
    return cleaned.strip()


def fix_common_generation_typos(text):
    """Correct recurring Khmer term corruption from local generation.

    These strings are not present in the corpus; they are model-side spelling
    drift when English insurance class labels are rewritten into Khmer. Keep
    this narrow so real source text is not broadly rewritten.
    """
    replacements = {
        "ធានរបងខាងតូច": "ធានារ៉ាប់រងខ្នាតតូច",
        "ធានរបងឯកជន": "ធានារ៉ាប់រងអាយុជីវិត",
        "ធានរបងទូទៅ": "ធានារ៉ាប់រងទូទៅ",
        "ធានរបង": "ធានារ៉ាប់រង",
    }
    for wrong, right in replacements.items():
        text = text.replace(wrong, right)
    return text


def clean_context_text_for_prompt(text):
    """Strip retrieval metadata/UI artifacts before the generator sees context."""
    lines = []
    for raw_line in (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            if lines and lines[-1]:
                lines.append("")
            continue
        if CONTEXT_SOURCE_NOISE_PATTERN.search(line):
            continue
        if HTML_ARTIFACT_PATTERN.search(line):
            continue
        line = CONTEXT_URL_PATTERN.sub("", line).strip()
        if not line:
            continue
        lines.append(line)
    cleaned = "\n".join(lines).strip()
    return re.sub(r"\n{3,}", "\n\n", cleaned)


def has_repetition_loop(text):
    """Detect runaway repeated spans while streaming."""
    compact = re.sub(r"\s+", " ", normalize_answer_text(text, final=False))
    if len(compact) < 220:
        return False
    for size in (45, 65, 90, 120):
        if len(compact) < size * 4:
            continue
        tail = compact[-size:]
        if len(tail.strip()) >= size * 0.8 and compact.count(tail) >= 4:
            return True
    sentences = [s.strip() for s in re.split(r"[។៕!?]", compact) if len(s.strip()) > 35]
    return len(sentences) >= 5 and len(set(sentences[-5:])) <= 2


LIST_MARKER_PATTERN = re.compile(r"^(?:[០-៩0-9]+\s*[.)៖]|[ក-អ]\s*[.)]|មាត្រា|ប្រការ|[-•])")


def clean_provision_text(text):
    """Fold PDF line-wrap newlines inside the verbatim provision text into
    spaces, while preserving the breaks that begin a heading or list item, so
    the deterministic provision answer reads as clean paragraphs/lists instead
    of one hard break per wrapped line."""
    out = []
    force_break = False  # a blank line was seen -- don't fold across it
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            # Blank lines in the corpus are stray PDF artifacts (e.g. one sits
            # mid-list before "៤." in laws មាត្រា ៣). Don't emit a visible empty
            # line, but do stop the next line from folding back up.
            force_break = True
            continue
        # Fold this line into the previous one (a mid-sentence wrap) unless it
        # opens a new structural block (heading / numbered / lettered / dash
        # item) or a blank line just separated them.
        if out and not force_break and not LIST_MARKER_PATTERN.match(line):
            out[-1] = out[-1] + " " + line
        else:
            out.append(line)
        force_break = False
    return "\n".join(out).strip()


SOURCE_NOISE_PATTERN = re.compile(
    r"^(?:Source file:|OCR method:|\[Page \d+\]|https?://|data-tip=|aria-label=)",
    re.IGNORECASE,
)


def clean_reference_text(text):
    """Prepare retrieved source text for direct, faithful display."""
    lines = []
    for raw in (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw.strip()
        if not line or SOURCE_NOISE_PATTERN.search(line):
            continue
        if HTML_ARTIFACT_PATTERN.search(line):
            break
        lines.append(line)
    cleaned = clean_provision_text("\n".join(lines))
    cleaned = normalize_answer_text(cleaned)
    return cleaned
