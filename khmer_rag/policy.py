"""Deterministic scope checks before retrieval or generation."""

import re

SCOPE_VIOLATION_PATTERNS = [
    # "(please) recommend a company/product/plan" -- an imperative
    # recommendation request directed AT the assistant. Deliberately
    # requires "ណែនាំ" to co-occur near "ក្រុមហ៊ុន/ផលិតផល/គម្រោង" rather than
    # triggering on "ណែនាំ" alone, since that word also appears constantly
    # in this corpus's own document titles ("គោលការណ៍ណែនាំ" = "guidance").
    re.compile(r"(សូម)?ណែនាំ.{0,30}(ក្រុមហ៊ុន|ផលិតផល|គម្រោង)"),
    # "which company/product is the best / better"
    re.compile(r"(ក្រុមហ៊ុន|ផលិតផល|គម្រោង).{0,30}ល្អ(បំផុត|ជាង)"),
    re.compile(r"មួយណា.{0,15}ល្អ(ជាង|បំផុត)?"),
    # Popularity / ranking / reputation framing -- functionally the same as
    # "which is best" but without the literal word "ល្អបំផុត"/"ណែនាំ", so the
    # patterns above miss it (e.g. "which company is most commonly used",
    # "which company has the best reputation"). Each still requires a
    # company/product/plan word to co-occur, same precision trade-off as
    # the patterns above.
    re.compile(r"(ក្រុមហ៊ុន|ផលិតផល|គម្រោង).{0,40}និយម.{0,15}ជាងគេ"),
    re.compile(r"(ក្រុមហ៊ុន|ផលិតផល|គម្រោង).{0,40}ពេញនិយម"),
    re.compile(r"(ក្រុមហ៊ុន|ផលិតផល|គម្រោង).{0,40}ល្បី.{0,15}ជាងគេ"),
    re.compile(r"(ក្រុមហ៊ុន|ផលិតផល|គម្រោង).{0,40}កេរ្តិ៍ឈ្មោះ.{0,10}ល្អ"),
    # "should I buy/choose/use from which company"
    re.compile(r"គួរ.{0,15}(ទិញ|ជ្រើសរើស|ប្រើ).{0,30}ក្រុមហ៊ុន"),
    # First-person "what should I choose/buy/use/look for" -- personal
    # product-selection advice framing, WITHOUT requiring a company to
    # co-occur (catches generic phrasing like "what should I choose?").
    # Deliberately anchored on literal "ខ្ញុំគួរ" ("I should") rather than
    # "គួរ" alone, since the corpus answers regulatory facts in third
    # person ("the insured may...", "the company must..."), so this is
    # unlikely to false-positive on legitimate retrievable content.
    #
    # The verb list is deliberately narrower than "any verb" -- it's
    # scoped to product-SELECTION verbs (choose/buy/use/look-at), not
    # general "what should I do" framing. A future claims-process corpus
    # (Internship Proposal §4's "step-by-step guidance on how to prepare
    # a claim" sub-domain) would legitimately need to answer first-person
    # "what should I do to file a claim" questions, which this pattern is
    # intentionally narrow enough not to catch.
    re.compile(r"ខ្ញុំគួរ.{0,20}(ជ្រើសរើស|ទិញ|ប្រើ|មើល)"),
    # "I don't want personal advice, but..." -- saying this is itself a
    # strong signal the question is advice-shaped, even when the user
    # explicitly disclaims wanting advice while still asking for it.
    re.compile(r"មិន.{0,10}(សុំ|ចង់បាន).{0,10}ប្រឹក្សា"),
    # "if you were me" -- explicitly asking for the assistant's own personal
    # pick. This phrasing essentially never appears in regulatory text, so
    # it's safe as a standalone trigger with no co-occurrence requirement.
    re.compile(r"បើអ្នកជាខ្ញុំ"),
    # "how much must I personally pay/spend" -- a personal monetary
    # calculation request, as opposed to a question about a fee amount
    # that IS stated as a fact in the corpus (e.g. "what is the annual
    # license fee" is fine; "how much must I pay per month" is not, since
    # nothing in the corpus states a personal monthly figure to retrieve).
    re.compile(r"ត្រូវ(បង់|ចំណាយ|ខាត).{0,10}ប៉ុន្មាន"),
    # Personal/product premium questions stay out of scope; official
    # statistics questions are allowed by is_scope_violation().
    re.compile(r"(បុព្វលាភ|ថ្លៃ|តម្លៃ).{0,30}ប៉ុន្មាន"),
    # First-person "should I <do X>, yes or no?" -- a personal decision the
    # user wants the assistant to make for them (e.g. "should I file a claim
    # now?", "should I renew?"). Anchored on "ខ្ញុំគួរ" ("I should") closing
    # with a yes/no marker ("ទេ"/"ឬទេ"), so it stays first-person and
    # decision-shaped. The corpus answers regulatory facts in third person and
    # never contains "ខ្ញុំគួរ", so this does not false-positive on legitimate
    # retrievable content. Broader than the product-SELECTION verb pattern
    # above, which only catches choose/buy/use/look-at.
    re.compile(r"ខ្ញុំគួរ.{0,40}(ឬ)?ទេ"),
    # Off-task requests that are not Cambodian-insurance-law questions at all:
    # asking the assistant to translate a word/phrase. Translation is a
    # general-purpose task outside this assistant's scope; "បកប្រែ" does not
    # appear in legitimate questions about the corpus's content.
    re.compile(r"បកប្រែ"),
]

OFFICIAL_STATISTICS_PATTERN = re.compile(
    r"(ស្ថិតិ|statistics|ឆ្នាំ|ប្រចាំឆ្នាំ|២០២[០-៩]|202[0-9]|បុព្វលាភធានារ៉ាប់រងដុល)",
    re.IGNORECASE,
)


def is_scope_violation(text):
    """Deterministic, score-independent check for questions asking the
    assistant to do something explicitly out of scope per the Internship
    Proposal's §4/§11 boundaries: recommend or compare a specific company
    or product, give advice framed as a personal choice, or calculate a
    personal monetary amount. See the comment block above for why this is
    a separate gate rather than relying on prompt instructions alone."""
    for pattern in SCOPE_VIOLATION_PATTERNS:
        if not pattern.search(text):
            continue
        if (
            pattern.pattern == r"(បុព្វលាភ|ថ្លៃ|តម្លៃ).{0,30}ប៉ុន្មាន"
            and OFFICIAL_STATISTICS_PATTERN.search(text)
        ):
            continue
        return True
    return False
