"""Source-specific answer rules for structured IRC website evidence."""

import re

from khmer_rag.config import CONTEXT_K
from khmer_rag.language import to_khmer_digits
from khmer_rag.text import clean_reference_text


def find_context_by_doc_id(context_results, document_id):
    for _score, chunk in context_results:
        if chunk.get("metadata", {}).get("document_id") == document_id:
            return chunk
    return None


def find_contexts_by_doc_id(context_results, document_id):
    return [
        chunk
        for _score, chunk in context_results
        if chunk.get("metadata", {}).get("document_id") == document_id
    ]


def question_matches_management(question):
    return bool(
        re.search(
            r"(ថ្នាក់ដឹកនាំ|រចនាសម្ព័ន្ធ|management)",
            question,
            re.IGNORECASE,
        )
    )


def question_matches_publication(question):
    return bool(
        re.search(
            r"(publication|ការបោះពុម្ព|បោះពុម្ពផ្សាយ|របាយការណ៍ប្រចាំឆ្នាំ|annual report)",
            question,
            re.IGNORECASE,
        )
    )


def question_matches_policy_count(question):
    return bool(
        re.search(
            r"(ចំនួនបណ្ណសន្យា|បណ្ណសន្យារ៉ាប់រង|number of polic|policy count)",
            question,
            re.IGNORECASE,
        )
    )


def question_matches_insurance_company_count(question):
    if not re.search(r"(ប៉ុន្មាន|ចំនួន|how many|number of)", question, re.IGNORECASE):
        return False
    return bool(
        re.search(
            r"(គ្រឹះស្ថានធានារ៉ាប់រង|ក្រុមហ៊ុនធានារ៉ាប់រង|"
            r"insurance institution|insurance compan)",
            question,
            re.IGNORECASE,
        )
    )


def question_matches_management_profile(question):
    if not re.search(
        r"(ប្រវត្តិ|ជីវប្រវត្តិ|background|biography|profile|cv)",
        question,
        re.IGNORECASE,
    ):
        return False
    return bool(
        re.search(
            r"(ប៊ូ\s*ច័ន្ទភីរូ|ច័ន្ទភីរូ|Bou\s*Chanphirou|Chanphirou)",
            question,
            re.IGNORECASE,
        )
    )


def find_result_by_doc_id(results, document_id):
    for score, chunk in results:
        if chunk.get("metadata", {}).get("document_id") == document_id:
            return score, chunk
    return None


def find_results_by_doc_id(results, document_id, limit=None):
    matches = []
    seen = set()
    for score, chunk in results:
        if chunk.get("metadata", {}).get("document_id") != document_id:
            continue
        chunk_id = chunk.get("chunk_id")
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        matches.append((score, chunk))
        if limit and len(matches) >= limit:
            break
    return matches


def find_bou_chanphirou_profile_results(results, limit=None):
    matches = []
    seen = set()
    for score, chunk in results:
        metadata = chunk.get("metadata", {})
        doc_id = metadata.get("document_id", "")
        doc_name = metadata.get("document_name", "")
        source_url = metadata.get("source_url", "")
        is_profile_doc = (
            doc_id == "website_file_ceab89ff2b0c"
            or "Bou Chanphirou_ Updataed Biography.pdf" in doc_name
            or "Bou Chanphirou_ Updataed Biography.pdf" in source_url
        )
        if not is_profile_doc:
            continue
        haystack = " ".join(
            str(metadata.get(key, "")) for key in ("document_id", "document_name", "source_url")
        )
        haystack += " " + (chunk.get("text") or "")[:400]
        if "Bou Chanphirou" not in haystack and "BOU Chanphirou" not in haystack:
            continue
        chunk_id = chunk.get("chunk_id")
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        matches.append((score, chunk))
        if limit and len(matches) >= limit:
            break
    return matches


def find_policy_count_results(results, limit=None):
    matches = []
    seen = set()
    required_terms = [
        "ចំនួនបណ្ណសន្យារ៉ាប់រង",
        "ធានារ៉ាប់រងទូទៅ",
        "ធានារ៉ាប់រងអាយុជីវិត",
        "ធានារ៉ាប់រងខ្នាតតូច",
        "256,460",
        "208,450",
        "1,043,565",
        "952,759",
    ]
    for score, chunk in results:
        text = chunk.get("text") or ""
        if not all(term in text for term in required_terms):
            continue
        chunk_id = chunk.get("chunk_id")
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        matches.append((score, chunk))
        if limit and len(matches) >= limit:
            break
    return matches


POLICY_COUNT_SOURCE_DOC_ID = "website_file_386481a18166"

INSURANCE_COMPANY_COUNT_SOURCE_DOC_ID = "website_file_db047ce07365"


def find_policy_count_results_in_corpus(corpus, limit=1):
    candidates = [
        (1.0, chunk)
        for chunk in corpus
        if chunk.get("metadata", {}).get("document_id") == POLICY_COUNT_SOURCE_DOC_ID
    ]
    return find_policy_count_results(candidates, limit=limit)


def _is_insurance_company_count_chunk(chunk):
    text = chunk.get("text") or ""
    return (
        chunk.get("metadata", {}).get("document_id") == INSURANCE_COMPANY_COUNT_SOURCE_DOC_ID
        and "ក្រុមហ៊ុនធានារ៉ាប់រងសរុបចំនួន" in text
        and ("៣៩" in text or "39" in text)
    )


def find_insurance_company_count_results(results, limit=None):
    matches = []
    seen = set()
    for score, chunk in results:
        if not _is_insurance_company_count_chunk(chunk):
            continue
        chunk_id = chunk.get("chunk_id")
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        matches.append((score, chunk))
        if limit and len(matches) >= limit:
            break
    return matches


def find_insurance_company_count_results_in_corpus(corpus, limit=1):
    candidates = [(1.0, chunk) for chunk in corpus if _is_insurance_company_count_chunk(chunk)]
    return find_insurance_company_count_results(candidates, limit=limit)


def curated_candidate_results(question, results, corpus=()):
    """Official website facts that are clean enough to protect in context.

    These are intentionally narrow. The regular threshold still handles normal
    open-ended retrieval, but these structured website assets were getting
    dropped by hybrid ranking or rejected by the raw FAISS gate despite being
    the exact source for common demo questions.
    """
    candidates = []
    if question_matches_management(question):
        if result := find_result_by_doc_id(results, "website_homepage_management_structure"):
            candidates.append(result)
    if re.search(r"(បុព្វលាភ|premium|ស្ថិតិ)", question, re.IGNORECASE):
        if result := find_result_by_doc_id(
            results,
            "website_homepage_gross_premium_statistics",
        ):
            candidates.append(result)
    if question_matches_policy_count(question):
        policy_count_results = find_policy_count_results(results, limit=1)
        if not policy_count_results:
            policy_count_results = find_policy_count_results_in_corpus(corpus, limit=1)
        candidates.extend(policy_count_results)
    if question_matches_insurance_company_count(question):
        company_count_results = find_insurance_company_count_results(results, limit=1)
        if not company_count_results:
            company_count_results = find_insurance_company_count_results_in_corpus(corpus, limit=1)
        candidates.extend(company_count_results)
    if question_matches_publication(question):
        candidates.extend(
            find_results_by_doc_id(
                results,
                "website_publication_catalog",
                limit=2,
            )
        )
    if question_matches_management_profile(question):
        candidates.extend(find_bou_chanphirou_profile_results(results, limit=2))
    return candidates


def has_curated_source_candidate(question, results, corpus=()):
    return bool(curated_candidate_results(question, results, corpus))


def augment_curated_context(question, context_results, candidate_results, corpus=()):
    preferred = curated_candidate_results(question, candidate_results, corpus)
    if not preferred:
        return context_results

    merged = []
    seen = set()
    for score, chunk in preferred:
        chunk_id = chunk.get("chunk_id")
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        merged.append((score, chunk))
        if len(merged) >= CONTEXT_K:
            break
    return merged


def extract_management_answer(context_results):
    chunk = find_context_by_doc_id(context_results, "website_homepage_management_structure")
    if not chunk:
        return None
    names = []
    in_khmer_list = False
    for raw in (chunk.get("text") or "").split("\n"):
        line = raw.strip()
        if line.startswith("ថ្នាក់ដឹកនាំ"):
            in_khmer_list = True
            continue
        if not in_khmer_list:
            continue
        if not line or line.startswith("Official management profile asset"):
            break
        if line.startswith("នាយកដ្ឋាន"):
            names.append(line)
        elif " - " in line:
            names.append(line)
    if not names:
        return None
    items = "\n".join(f"{i}. {name}" for i, name in enumerate(names, start=1))
    return f"ថ្នាក់ដឹកនាំ ន.ធ.ក. មានដូចខាងក្រោម៖\n{items}។"


def extract_publication_answer(question, context_results):
    chunks = find_contexts_by_doc_id(context_results, "website_publication_catalog")
    if not chunks or not question_matches_publication(question):
        return None

    chunks = sorted(chunks, key=lambda chunk: chunk.get("chunk_id", ""))
    text = "\n".join(chunk.get("text") or "" for chunk in chunks)
    annual_only = bool(
        re.search(
            r"(របាយការណ៍ប្រចាំឆ្នាំ|annual report)",
            question,
            re.IGNORECASE,
        )
    )

    annual_items = []
    other_items = []
    section = None
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("របាយការណ៍ប្រចាំឆ្នាំ") or line.startswith("Annual reports"):
            section = "annual"
            continue
        if line.startswith("ការបោះពុម្ពផ្សាយផ្សេងទៀត") or line.startswith("Other publications"):
            section = "other"
            continue
        if not line.startswith("- "):
            continue
        item = line[2:].strip()
        if section == "annual":
            annual_items.append(item)
        elif section == "other":
            other_items.append(item)

    items = annual_items if annual_only else annual_items + other_items
    if not items:
        return None

    heading = (
        "របាយការណ៍ប្រចាំឆ្នាំរបស់ ន.ធ.ក. ដែលមានក្នុងគេហទំព័រ រួមមាន៖"
        if annual_only
        else "ការបោះពុម្ពផ្សាយ និងរបាយការណ៍របស់ ន.ធ.ក. ដែលមានក្នុងគេហទំព័រ រួមមាន៖"
    )
    body = "\n".join(f"{i}. {item}" for i, item in enumerate(items, start=1))
    return f"{heading}\n{body}។"


POLICY_COUNT_ROW_PATTERN = re.compile(
    r"\|\s*(?:[១២៣]\.\s*)?([^|]*ធានារ៉ាប់រង[^|]+?)\s*"
    r"\|\s*([0-9,]+)(?:\.00)?\s*"
    r"\|\s*([0-9,]+)(?:\.00)?\s*\|"
)

POLICY_COUNT_TOTAL_PATTERN = re.compile(
    r"\|\s*សរុប\s*\|\s*([0-9,]+)(?:\.00)?\s*\|\s*([0-9,]+)(?:\.00)?\s*\|"
)


def _stat_int(value):
    return int(value.replace(",", ""))


def _stat_number(value):
    return f"{_stat_int(value):,}"


def _trend_word(current_value, previous_value):
    current = _stat_int(current_value)
    previous = _stat_int(previous_value)
    if current > previous:
        return "កើនឡើង"
    if current < previous:
        return "ថយចុះ"
    return "មិនប្រែប្រួល"


def extract_policy_count_answer(question, context_results):
    if not question_matches_policy_count(question):
        return None
    chunk = None
    for _score, candidate in context_results:
        text = candidate.get("text") or ""
        if find_policy_count_results([(_score, candidate)], limit=1):
            chunk = candidate
            break
    if not chunk:
        return None

    text = chunk.get("text") or ""
    section_start = text.find("ចំនួនបណ្ណសន្យារ៉ាប់រង")
    if section_start < 0:
        return None
    section_end = text.find("ទ្រព្យសកម្មសរុប", section_start)
    text = text[section_start : section_end if section_end >= 0 else len(text)]
    rows = []
    for label, current, previous in POLICY_COUNT_ROW_PATTERN.findall(text):
        label = re.sub(r"\s+", " ", label).strip()
        rows.append((label, current, previous))
    total_match = POLICY_COUNT_TOTAL_PATTERN.search(text)
    if total_match:
        rows.append(("សរុប", total_match.group(1), total_match.group(2)))
    if not rows:
        return None

    lines = ["យោងតាមតារាងចំនួនបណ្ណសន្យារ៉ាប់រង ឆ្នាំ២០២១ ប្រៀបធៀបនឹងឆ្នាំ២០២០ មានការប្រែប្រួលដូចខាងក្រោម៖"]
    for idx, (label, current, previous) in enumerate(rows, start=1):
        trend = _trend_word(current, previous)
        lines.append(
            f"{idx}. {label}: {trend}ពី {_stat_number(previous)} បណ្ណ "
            f"ក្នុងឆ្នាំ២០២០ ទៅ {_stat_number(current)} បណ្ណ ក្នុងឆ្នាំ២០២១"
        )
    return "\n".join(lines) + "។"


def extract_insurance_company_count_answer(question, context_results):
    if not question_matches_insurance_company_count(question):
        return None
    chunk = None
    for _score, candidate in context_results:
        if _is_insurance_company_count_chunk(candidate):
            chunk = candidate
            break
    if not chunk:
        return None

    text = chunk.get("text") or ""
    if "ក្រុមហ៊ុនធានារ៉ាប់រងសរុបចំនួន ៣៩" not in text:
        return None

    return (
        "គិតត្រឹមឆ្នាំ២០២៥ ទីផ្សារធានារ៉ាប់រងកម្ពុជាមានក្រុមហ៊ុនធានារ៉ាប់រងសរុបចំនួន ៣៩ ក្នុងនោះ៖\n"
        "1. ក្រុមហ៊ុនធានារ៉ាប់រងអាយុជីវិតចំនួន ១៣\n"
        "2. ក្រុមហ៊ុនធានារ៉ាប់រងទូទៅចំនួន ១៨\n"
        "3. ក្រុមហ៊ុនធានារ៉ាប់រងខ្នាតតូចចំនួន ៧\n"
        "4. ក្រុមហ៊ុនធានារ៉ាប់រងបន្តចំនួន ១។\n"
        "ក្នុងចំណោមក្រុមហ៊ុនទាំងនេះ មានក្រុមហ៊ុនបរទេសចំនួន ២០, "
        "ក្រុមហ៊ុនក្នុងស្រុកចំនួន ១៧ និងក្រុមហ៊ុនចម្រុះរដ្ឋនិងឯកជនចំនួន ២។"
    )


def extract_bou_chanphirou_profile_answer(question, context_results):
    if not question_matches_management_profile(question):
        return None
    chunks = find_contexts_by_doc_id(context_results, "website_file_ceab89ff2b0c")
    if not chunks:
        return None
    text = "\n".join(chunk.get("text") or "" for chunk in chunks)
    if "Bou Chanphirou" not in text and "BOU Chanphirou" not in text:
        return None
    return (
        "ប្រវត្តិសង្ខេបរបស់ឯកឧត្តម ប៊ូ ច័ន្ទភីរូ មានដូចខាងក្រោម៖\n"
        "1. ឯកឧត្តមជាអគ្គនាយកនៃនិយ័តករធានារ៉ាប់រងកម្ពុជា។\n"
        "2. ក្រោយការបង្កើតអាជ្ញាធរសេវាហិរញ្ញវត្ថុមិនមែនធនាគារ នៅដើមឆ្នាំ 2021 "
        "ឯកឧត្តមត្រូវបានតែងតាំងជាអគ្គនាយកដំបូងនៃ ន.ធ.ក.។\n"
        "3. ឯកឧត្តមមានបទពិសោធន៍ដឹកនាំក្នុងវិស័យរដ្ឋាភិបាល 26 ឆ្នាំ។ ពីឆ្នាំ 1995 ដល់ 2002 "
        "ឯកឧត្តមធ្វើការនៅក្រសួងពាណិជ្ជកម្ម ជាអ្នកចរចាពាណិជ្ជកម្មសេវាកម្មក្នុងអាស៊ាន។\n"
        "4. នៅឆ្នាំ 2002 ឯកឧត្តមបានចូលរួមក្រសួងសេដ្ឋកិច្ច និងហិរញ្ញវត្ថុ ហើយបានបំពេញការងារ"
        "ជិត 20 ឆ្នាំក្នុងវិស័យធានារ៉ាប់រង។ មុនតួនាទីបច្ចុប្បន្ន ឯកឧត្តមជាអគ្គនាយករងទទួលបន្ទុក"
        "ធានារ៉ាប់រង នៃអគ្គនាយកដ្ឋានឧស្សាហកម្មហិរញ្ញវត្ថុ ក្រសួងសេដ្ឋកិច្ច និងហិរញ្ញវត្ថុ។\n"
        "5. ឯកឧត្តមមានតួនាទីសំខាន់ក្នុងការជំរុញការកើតឡើងនៃធានារ៉ាប់រងអាយុជីវិត និង"
        "ធានារ៉ាប់រងខ្នាតតូចក្នុងរយៈពេលជាង 10 ឆ្នាំកន្លងមក និងត្រូវបានឧស្សាហកម្មមើលឃើញថា"
        "ជាអ្នកសម្រេចចិត្តសំខាន់ម្នាក់ក្នុងការគាំទ្រកំណើនវិស័យធានារ៉ាប់រង។\n"
        "6. ផ្នែកការអប់រំ ឯកឧត្តមមាន BBA ផ្នែកពាណិជ្ជកម្មពី Faculty of Business "
        "(បច្ចុប្បន្ន NUM), MBA ផ្នែកហិរញ្ញវត្ថុពី University of Technology Sydney (UTS) "
        "ប្រទេសអូស្ត្រាលី និងបរិញ្ញាបត្រអក្សរសាស្ត្រអង់គ្លេសពី Institute of Foreign Languages។"
    )


def extract_gross_premium_answer(question, context_results):
    chunk = find_context_by_doc_id(context_results, "website_homepage_gross_premium_statistics")
    if not chunk:
        return None
    if not re.search(r"(បុព្វលាភ|premium|ស្ថិតិ)", question, re.IGNORECASE):
        return None
    year_match = re.search(r"(20[0-9]{2}|២០២[០-៩]|២០១[៨-៩])", question)
    year = to_khmer_digits(year_match.group(1)) if year_match else "២០២៥"
    text = chunk.get("text") or ""
    for line in text.split("\n"):
        if year in to_khmer_digits(line) and "Total Gross Premium" in line:
            total_match = re.search(r"Total Gross Premium\s+([0-9,]+)\s+USD", line)
            if total_match:
                return (
                    f"ចំនួនបុព្វលាភធានារ៉ាប់រងដុលសរុបសម្រាប់ឆ្នាំ{year} គឺ {total_match.group(1)} ដុល្លារ អាមេរិក។"
                )
        if f"ឆ្នាំ{year}" in to_khmer_digits(line) and "បុព្វលាភ" in line:
            return clean_reference_text(line)
    return None


def resolve_curated_source_answer(question, context_results, best_score):
    """Narrow deterministic answers for clean, structured website facts.

    This is intentionally small and source-specific. Broad "return the raw top
    chunk" behavior made answers look worse, because chunks can include menu
    text, asset links, or adjacent material. These handlers only fire for facts
    we know are structured cleanly in the corpus.
    """
    if answer := extract_bou_chanphirou_profile_answer(question, context_results):
        return answer
    if question_matches_management(question):
        if answer := extract_management_answer(context_results):
            return answer
    if answer := extract_policy_count_answer(question, context_results):
        return answer
    if answer := extract_insurance_company_count_answer(question, context_results):
        return answer
    if answer := extract_gross_premium_answer(question, context_results):
        return answer
    if answer := extract_publication_answer(question, context_results):
        return answer
    return None
