"""Document identifiers and source category labels."""

import re

from khmer_rag.categories import WEBSITE_CATEGORY_LABEL_KH

_CHUNK_SUFFIX_RE = re.compile(r"_chunk_\d+$")

SOURCE_PDF_DIRS = {
    "law": "laws",
    "prakas": "prakas",
    "sub_decree": "sub_decrees",
    "guidance": "guidance",
}

DOCUMENT_TYPE_KH = {
    "law": "ច្បាប់",
    "prakas": "ប្រកាស",
    "guidance": "គោលការណ៍ណែនាំ",
    "sub_decree": "អនុក្រឹត្យ",
    "website": "ទំព័រគេហទំព័រ",
}


DOCUMENT_TYPE_KH.update(WEBSITE_CATEGORY_LABEL_KH)


def document_id_for_chunk(chunk_id):
    """ "law_04_chunk_9" -> "law_04"."""
    return _CHUNK_SUFFIX_RE.sub("", str(chunk_id))
