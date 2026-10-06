"""Khmer language detection, greetings, and lexical tokenization."""

import re

from khmer_rag.config import NGRAM_SIZE

DIGIT_RUN_PATTERN = re.compile(r"[០១២៣៤៥៦៧៨៩0-9]+")

ARTICLE_REF_PATTERN = re.compile(r"(មាត្រា|ប្រការ)\s*([០១២៣៤៥៦៧៨៩0-9]+)")

_ARABIC_TO_KHMER = str.maketrans("0123456789", "០១២៣៤៥៦៧៨៩")


def to_khmer_digits(s):
    return s.translate(_ARABIC_TO_KHMER)


ARTICLE_REF_BOOST = 20

KHMER_SCRIPT_PATTERN = re.compile(r"[\u1780-\u17FF]")


def contains_khmer_script(text):
    return bool(KHMER_SCRIPT_PATTERN.search(text))


def khmer_aware_tokenize(text):
    """Tokenize Khmer with character trigrams and Latin words as whole tokens.

    Boost compound provision references and normalize their digits so queries
    such as មាត្រា 10 match corpus headings written as មាត្រា ១០."""
    tokens = []
    # Compound provision-reference tokens first (see ARTICLE_REF_PATTERN). Done
    # over the whole text, not per-word, since the keyword ("មាត្រា"/"ប្រការ")
    # and its number sit in separate whitespace tokens in the corpus heading
    # ("មាត្រា ១០"). Keeping the keyword in the compound prevents a "មាត្រា ៥"
    # query from matching a "ប្រការ ៥" heading in another document.
    for keyword, ref_num in ARTICLE_REF_PATTERN.findall(text):
        tokens.extend([keyword + to_khmer_digits(ref_num)] * ARTICLE_REF_BOOST)
    for word in text.split():
        if KHMER_SCRIPT_PATTERN.search(word):
            # Khmer is unspaced, so an entire question collapses into one
            # whitespace-token. Article numbers ("មាត្រា១០") then get glued to
            # surrounding script and shredded into n-grams, so the
            # distinguishing digit run never survives as a standalone token to
            # match the corpus -- where "មាត្រា ១០" keeps "១០" separate. Pull
            # every maximal digit run (Khmer ០-៩ U+17E0-17E9 or ASCII 0-9) out
            # as its own token first, so the article number matches lexically
            # regardless of spacing on either side.
            for digit_run in DIGIT_RUN_PATTERN.findall(word):
                tokens.append(digit_run)
            chars = list(word)
            if len(chars) < NGRAM_SIZE:
                tokens.append(word)
            else:
                for i in range(len(chars) - NGRAM_SIZE + 1):
                    tokens.append("".join(chars[i : i + NGRAM_SIZE]))
        else:
            tokens.append(word.lower())
    return tokens


DISCLAIMER_TEXT = (
    "ចំណាំ៖ ចម្លើយនេះមានគោលបំណងផ្តល់ព័ត៌មានទូទៅសម្រាប់ការអប់រំប៉ុណ្ណោះ "
    "មិនមែនជាការផ្តល់ប្រឹក្សាផ្នែកហិរញ្ញវត្ថុ ឬច្បាប់ផ្ទាល់ខ្លួនទេ។ "
    "សម្រាប់ករណីជាក់លាក់របស់អ្នក សូមទាក់ទងភ្នាក់ងារធានារ៉ាប់រងដែលមានអាជ្ញាប័ណ្ណ "
    "ឬនិយ័តករធានារ៉ាប់រងកម្ពុជា (IRC)។"
)

REDIRECT_TEXT = "សូមទាក់ទងភ្នាក់ងារធានារ៉ាប់រងដែលមានអាជ្ញាប័ណ្ណ ឬនិយ័តករធានារ៉ាប់រងកម្ពុជា (IRC) សម្រាប់ជំនួយបន្ថែម។"

NOT_KHMER_QUESTION_TEXT = "សូមសួរសំណួរជាភាសាខ្មែរដែលពាក់ព័ន្ធនឹងធានារ៉ាប់រងកម្ពុជា"

GREETING_RESPONSE_TEXT = (
    "ជម្រាបសួរ! ខ្ញុំជាជំនួយការធានារ៉ាប់រងកម្ពុជា។ តើលោក/លោកស្រី មានចម្ងល់អ្វីខ្លះដែរទាក់ទងនឹងការធានារ៉ាប់រង?"
)

KHMER_GREETING_TERMS = {
    "សួស្តី",
    "សួស្ដី",
    "ជម្រាបសួរ",
    "ជំរាបសួរ",
    "សូមជម្រាបសួរ",
    "សូមជំរាបសួរ",
    "សូមគោរពជម្រាបសួរ",
    "សូមគោរពជំរាបសួរ",
    "អរុណសួស្តី",
    "អរុណសួស្ដី",
    "ទិវាសួស្តី",
    "ទិវាសួស្ដី",
    "សាយណ្ហសួស្តី",
    "សាយណ្ហសួស្ដី",
    "រាត្រីសួស្តី",
    "រាត្រីសួស្ដី",
    "ស្វាគមន៍",
}

GREETING_POLITE_SUFFIXES = {"", "បាទ", "ចាស", "ចា", "លោក", "លោកស្រី"}


def normalize_greeting_text(text):
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text or "")
    return re.sub(r"[\s!?,.។៕:;\"'«»“”‘’()\[\]{}<>…-]+", "", text)


NORMALIZED_KHMER_GREETING_TERMS = {normalize_greeting_text(term) for term in KHMER_GREETING_TERMS}


def is_khmer_greeting(text):
    normalized = normalize_greeting_text(text)
    if not normalized:
        return False
    if normalized in NORMALIZED_KHMER_GREETING_TERMS:
        return True
    for term in NORMALIZED_KHMER_GREETING_TERMS:
        if normalized.startswith(term) and normalized[len(term) :] in GREETING_POLITE_SUFFIXES:
            return True
    return False
