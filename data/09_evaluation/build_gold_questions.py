#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_gold_questions.py
=======================
Reusable generator for the AI-assisted half of each gold_questions.json
(question 26 onward). The first 25 questions of every category are written
BY HAND (0% AI) and are NEVER touched by this script.

WHY THIS EXISTS
---------------
Each gold Q&A must read like the manually-written first 25:
  * the ANSWER is a VERBATIM passage taken straight from the source chunk
    (no paraphrasing, no fabricated words/spaces), and
  * the QUESTION is phrased so the answer naturally echoes it, the way a real
    user (who does NOT know article/prakas/document reference codes) would ask.

So we DON'T let the model free-write answers. Instead we:
  1. pick a chunk,
  2. slice the exact passage that answers one question (a single paragraph,
     a definition, or a "lead sentence + its list"), and
  3. hand-write a natural question for that passage.

The slicing + cleaning is deterministic (this file). Only the QUESTION text and
the choice of passage are authored. That keeps every answer provably verbatim.

HOW TO ADD A NEW CATEGORY (laws / sub_decrees / prakas)
------------------------------------------------------
1. Read the chunk file, e.g. data/08_merged_chunked_documents/laws_chunks.json,
   and skim each chunk's `content`.
2. For each Q&A you want, append a tuple to SPECS["laws"]:
       (chunk_id, question, slice_mode)
3. Run:  python data/09_evaluation/build_gold_questions.py laws
   It rebuilds questions (N_HUMAN+1 .. end), validates, sorts by chunk in
   ascending document order, renumbers ids, and writes the file back.
   The first N_HUMAN questions are preserved byte-for-byte.

SLICE MODES (third item of each spec tuple)
-------------------------------------------
  ('full',)                  -> the whole cleaned chunk (use only when the whole
                                chunk is one tight passage and <= MAX_LEN).
  ('m', start, end)          -> verbatim text between two marker substrings.
                                start=None means "from the beginning",
                                end=None means "to the end". Use for prose
                                paragraphs / single sentences. Pick SHORT,
                                contiguous markers (avoid spanning a space that
                                might be a line-wrap join).
  ('h', n)                   -> lead line + the first n list-item lines. Use for
                                "lead sentence ending in : + bulleted list".
                                Robust against wrap/space ambiguity -> prefer
                                this for lists instead of ('m', ...).
  ('hs', startmarker, n)     -> like 'h' but the lead is the line CONTAINING
                                startmarker (not the chunk's first line). Use to
                                grab a sub-section/list that starts mid-chunk.

RULES THE VALIDATOR ENFORCES (so the output stays clean)
--------------------------------------------------------
  * answer length in [MIN_LEN, MAX_LEN]  (chatbot answer cap is 1024).
  * answer is a whitespace-normalized substring of the cleaned chunk (VERBATIM).
  * answer never starts with a bare "-" / "•" bullet (no orphan lists).
  * no reference codes in the QUESTION or the ANSWER
    (មាត្រា<digit>, លេខ ០.., /សណន, ឧ.ធ.ក, ច្បាប់ស្តីពី..លេខ) — users don't ask
    with codes, and the typo'd ឧ.ធ.ក must not leak.
  * no PDF artifacts left in the answer (ធានា-រ៉ាប់រង hyphenation, ថាល់ garble).
  * questions unique, and none duplicates a hand-written first-25 question.

NOTE ON guidance: the SPEC below is the real, shipped guidance set — keep it as
a worked reference when authoring the other three categories.
"""
import json, re, sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Per-category config: where the chunks live, where the gold file lives, how many
# leading questions are human-written (never regenerated), and the id prefix.
CONFIG = {
    "guidance": dict(
        chunks="data/08_merged_chunked_documents/guidance_chunks.json",
        gold="data/09_evaluation/guidance/gold_questions.json",
        n_human=25, prefix="guidance_qid_"),
    "laws": dict(
        chunks="data/08_merged_chunked_documents/laws_chunks.json",
        gold="data/09_evaluation/laws/gold_questions.json",
        n_human=25, prefix="law_qid_"),
    "sub_decrees": dict(
        chunks="data/08_merged_chunked_documents/sub_decrees_chunks.json",
        gold="data/09_evaluation/sub_decrees/gold_questions.json",
        n_human=25, prefix="sub_decree_qid_"),
    "prakas": dict(
        chunks="data/08_merged_chunked_documents/prakas_chunks.json",
        gold="data/09_evaluation/prakas/gold_questions.json",
        n_human=25, prefix="prakas_qid_"),
}

MIN_LEN, MAX_LEN = 80, 1024

# ---------------------------------------------------------------------------
# Deterministic cleaner: turns a raw chunk `content` into the readable, verbatim
# text we slice answers from. Mirrors how the live chatbot tidies a provision.
# ---------------------------------------------------------------------------
LIST_MARKER = re.compile(r'^(?:[០-៩0-9]+\s*[.)៖]|[ក-អ]\s*[-.)]|[-•])')

def clean(text):
    out = []
    force_break = False  # a blank line was seen -> don't fold across it
    for raw in text.split('\n'):
        line = raw.strip()
        if not line:
            force_break = True
            continue
        if out and not force_break and not LIST_MARKER.match(line):
            # continuation of a wrapped line: Khmer is continuous, so join with
            # NO space; drop a trailing PDF hyphen (ធានា-\nរ៉ាប់រង -> ធានារ៉ាប់រង)
            out[-1] = out[-1][:-1] + line if out[-1].endswith('-') else out[-1] + line
        else:
            out.append(line)
        force_break = False
    t = '\n'.join(out).strip()
    # de-hyphenate PDF word-break hyphens between Khmer letters (>=2 letters
    # before the hyphen, so single-letter list markers ក- ខ- are preserved)
    t = re.sub(r'(?<=[ក-៿][ក-៿])-(?=[ក-៿])', '', t)
    # restore the space after a sentence-final ។ when glued to the next sentence
    t = re.sub(r'។(?=[ក-៩A-Za-z“‘(])', '។ ', t)
    return t


def build_answer(byid, cid, mode):
    t = clean(byid[cid]['content'])
    lines = t.split('\n')
    if mode[0] == 'full':
        return t
    if mode[0] == 'm':
        _, s, e = mode
        i = 0 if s is None else t.index(s)
        j = len(t) if e is None else t.index(e) + len(e)
        return t[i:j].strip()
    if mode[0] in ('h', 'hs'):
        if mode[0] == 'h':
            si, nitems = 0, mode[1]
        else:
            sm, nitems = mode[1], mode[2]
            si = next(k for k, l in enumerate(lines) if sm in l)
        block = [lines[si]]
        cnt, k = 0, si + 1
        while k < len(lines) and cnt < nitems:
            block.append(lines[k])
            if LIST_MARKER.match(lines[k]):
                cnt += 1
            k += 1
        return '\n'.join(block).strip()
    raise ValueError(f"unknown slice mode {mode!r}")


def chunk_sort_key(chunk_id):
    # e.g. guidance_07_chunk_3 -> (7, 3); law_02_chunk_19 -> (2, 19)
    m = re.search(r'_(\d+)_chunk_(\d+)$', chunk_id)
    return (int(m.group(1)), int(m.group(2)))


# Guards used by the validator.
Q_CODE = re.compile(r'លេខ ០|/សណន|ឧ\.ធ\.ក|មាត្រា[០-៩]|ច្បាប់ស្តីពី.*លេខ')
A_CODE = re.compile(r'មាត្រា[០-៩]|លេខ ០|/សណន|ឧ\.ធ\.ក')


def build(category):
    cfg = CONFIG[category]
    S = SPECS.get(category, [])
    if not S:
        sys.exit(f"No SPEC for '{category}' yet — add tuples to SPECS['{category}'].")
    CH = json.load(open(os.path.join(ROOT, cfg["chunks"])))
    byid = {c['chunk_id']: c for c in CH}
    gold = json.load(open(os.path.join(ROOT, cfg["gold"])))
    n = cfg["n_human"]
    human = gold[:n]
    assert len(human) == n, f"expected {n} human questions, found {len(human)}"

    entries = []
    for cid, q, mode in S:
        if cid not in byid:
            sys.exit(f"chunk not found: {cid}")
        entries.append({"question_id": None, "question": q,
                        "expected_chunks": [cid],
                        "expected_answer": build_answer(byid, cid, mode)})

    # sort ascending by chunk (stable -> keeps order of multiple Q&A per chunk)
    entries.sort(key=lambda e: chunk_sort_key(e['expected_chunks'][0]))
    for i, e in enumerate(entries, n + 1):
        e['question_id'] = f"{cfg['prefix']}{i:03d}"

    # ---- validate ----
    nows = lambda s: re.sub(r'\s+', '', s)
    human_qs = {h['question'] for h in human}
    seen = set()
    for e in entries:
        a, cid, qid = e['expected_answer'], e['expected_chunks'][0], e['question_id']
        L = len(a)
        assert MIN_LEN <= L <= MAX_LEN, (qid, "length", L)
        assert nows(a) in nows(clean(byid[cid]['content'])), (qid, "NOT VERBATIM")
        assert not a.lstrip().startswith(('-', '•')), (qid, "orphan bullet")
        assert not A_CODE.search(a), (qid, "code/typo in answer")
        assert 'ធានា-រ៉ាប់រង' not in a and 'ថាល់' not in a, (qid, "PDF artifact")
        assert not Q_CODE.search(e['question']), (qid, "code in question")
        assert e['question'] not in human_qs, (qid, "duplicates a first-25 question")
        assert e['question'] not in seen, (qid, "duplicate question")
        seen.add(e['question'])

    out = human + entries
    path = os.path.join(ROOT, cfg["gold"])
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write('\n')
    lens = [len(e['expected_answer']) for e in entries]
    print(f"[{category}] wrote {len(out)} total ({n} human + {len(entries)} AI). "
          f"answer len min/max = {min(lens)}/{max(lens)}. first {n} untouched.")


# ===========================================================================
# SPECS — one list per category. (chunk_id, question, slice_mode)
# guidance is the shipped reference set. Fill the others the same way.
# ===========================================================================
SPECS = {
"guidance": [
("guidance_01_chunk_1","តើក្រុមហ៊ុនធានារ៉ាប់រងខ្នាតតូច ត្រូវចាត់បញ្ជូនបុគ្គលិកបែបណា ពេលមានហានិភ័យធានារ៉ាប់រងកើតឡើង?",('m',"ក្រុមហ៊ុនធានារ៉ាប់រងខ្នាតតូច ត្រូវចាត់បញ្ជូនបុគ្គលិក",None)),
("guidance_01_chunk_2","តើក្រុមហ៊ុនធានារ៉ាប់រងខ្នាតតូច ត្រូវពន្យល់អ្នកទាមទារសំណងធានារ៉ាប់រងអំពីអ្វី?",('m',"ក្រុមហ៊ុនធានារ៉ាប់រងខ្នាតតូច ត្រូវពន្យល់","ឱ្យបានច្បាស់លាស់។")),
("guidance_01_chunk_3","តើអ្នកទាមទារសំណងធានារ៉ាប់រងមានសិទ្ធិអ្វី ពេលក្រុមហ៊ុនសម្រេចបដិសេធ ឬធ្វើសំណងមិនពេញលេញ?",('m',"អ្នកទាមទារសំណងធានារ៉ាប់រងមានសិទ្ធិ",None)),
("guidance_01_chunk_3","តើក្រុមហ៊ុនធានារ៉ាប់រងខ្នាតតូច ត្រូវធ្វើដូចម្តេច ពេលការទាមទារសំណងមានភាពស្មុគស្មាញ ត្រូវការពេលបន្ថែម?",('m',"ក្រុមហ៊ុនធានារ៉ាប់រងខ្នាតតូច ត្រូវជូនដំណឹងមក","ត្រូវការពេលវេលាបន្ថែម។")),
("guidance_01_chunk_6","តើក្រុមហ៊ុនធានារ៉ាប់រងខ្នាតតូច ត្រូវរក្សាទុកឯកសារអំពីការទទួល និងដោះស្រាយសំណងធានារ៉ាប់រង ក្នុងរយៈពេលអប្បបរមាប៉ុន្មានឆ្នាំ?",('full',)),
("guidance_02_chunk_2","តើសមាគមជើងសាធានារ៉ាប់រងកម្ពុជា ត្រូវរៀបចំអ្វីខ្លះ មុននឹងផ្តល់វគ្គបណ្តុះបណ្តាលវិជ្ជាជីវៈជើងសាធានារ៉ាប់រង?",('full',)),
("guidance_02_chunk_4","តើសិក្ខាកាមដែលចាត់ទុកថាប្រឡងជាប់ ត្រូវមានកម្រិតពិន្ទុចាប់ពីប៉ុន្មានភាគរយ?",('m',None,"ឡើងទៅ។")),
("guidance_02_chunk_4","តើវិញ្ញាបនបត្រវិជ្ជាជីវៈជើងសាធានារ៉ាប់រង ត្រូវចុះហត្ថលេខាដោយអ្នកណា?",('m',"វិញ្ញាបនបត្រវិជ្ជាជីវៈជើងសាធានារ៉ាប់រង ត្រូវបានផ្តល់ជូន",None)),
("guidance_02_chunk_6","តើបុគ្គលវិជ្ជាជីវៈជើងសាធានារ៉ាប់រងដែលបានបញ្ចប់វគ្គបណ្តុះបណ្តាល ត្រូវបន្តចូលរួមវគ្គអភិវឌ្ឍវិជ្ជាជីវៈយ៉ាងតិចប៉ុន្មានម៉ោងក្នុងមួយឆ្នាំ?",('full',)),
("guidance_04_chunk_1","តើវិជ្ជាជីវៈជើងសាធានារ៉ាប់រង សំដៅដល់អ្វី?",('full',)),
("guidance_04_chunk_4","តើបុគ្គលិកក្រុមហ៊ុនជើងសាធានារ៉ាប់រងដែលមានវិជ្ជាជីវៈជើងសាធានារ៉ាប់រង ត្រូវមានចំណេះដឹងវិជ្ជាជីវៈអ្វីខ្លះ?",('full',)),
("guidance_04_chunk_5","តើបុគ្គលិកក្រុមហ៊ុនជើងសាធានារ៉ាប់រងដែលមានវិជ្ជាជីវៈជើងសាធានារ៉ាប់រង ត្រូវប្រកាន់យកនូវអាកប្បកិរិយាអ្វីខ្លះ ក្នុងផ្នែកសុចរិតភាព និងភាពស្មោះត្រង់?",('h',5)),
("guidance_04_chunk_5","តើក្រុមហ៊ុនជើងសាធានារ៉ាប់រង ត្រូវធានាយ៉ាងណាចំពោះអត្ថបទផ្សាយពាណិជ្ជកម្មដែលខ្លួនបានរៀបចំ?",('m',"ក្រុមហ៊ុនជើងសាធានារ៉ាប់រង ត្រូវធានាឱ្យបានថា",None)),
("guidance_04_chunk_9","តើបុគ្គលិកក្រុមហ៊ុនជើងសាធានារ៉ាប់រងដែលមានវិជ្ជាជីវៈជើងសាធានារ៉ាប់រង ត្រូវប្រកាន់យកនូវអាកប្បកិរិយាអ្វីខ្លះ ក្នុងការប្រតិបត្តិវិជ្ជាជីវៈ?",('h',5)),
("guidance_04_chunk_14","តើបុគ្គលវិជ្ជាជីវៈជើងសាធានារ៉ាប់រង ត្រូវប្រកាន់យកនូវអាកប្បកិរិយាអ្វីខ្លះ ក្នុងការបន្តបម្រើអតិថិជនបន្ទាប់ពីការចេញបណ្ណសន្យារ៉ាប់រង?",('h',3)),
("guidance_04_chunk_16","តើការបំពានក្រមសីលធម៌ត្រូវបានចែកជាប៉ុន្មានកម្រិត?",('m',"ការមិនអនុវត្តតាមគោលការណ៍ប្រតិបត្តិវិជ្ជាជីវៈខាងលើ","ត្រូវបានកំណត់ដូចខាងក្រោម៖")),
("guidance_04_chunk_18","តើក្រុមហ៊ុនជើងសាធានារ៉ាប់រង ត្រូវផ្តល់វគ្គបណ្តុះបណ្តាលអំពីក្រមសីលធម៌វិជ្ជាជីវៈជើងសាធានារ៉ាប់រងជូនបុគ្គលិកនៅពេលណា?",('full',)),
("guidance_05_chunk_1","តើកិច្ចសន្យាធានារ៉ាប់រងខ្នាតតូច គឺជាអ្វី?",('m',None,"ពីអ្នកដែលត្រូវបានធានារ៉ាប់រង។")),
("guidance_05_chunk_1","តើកិច្ចសន្យាធានារ៉ាប់រងខ្នាតតូចមួយ ត្រូវមានចំនួនទឹកប្រាក់អតិបរមាដែលត្រូវបានធានារ៉ាប់រងប៉ុន្មាន?",('m',"កិច្ចសន្យាធានារ៉ាប់រងខ្នាតតូចមួយ","មិនលើសពី ២០ ០០០ ០០០ (ម្ភៃលាន) រៀល។")),
("guidance_05_chunk_2","តើក្រុមហ៊ុនធានារ៉ាប់រងខ្នាតតូច ត្រូវចេញវិញ្ញាបនបត្រធានារ៉ាប់រង និងបណ្ណសន្យារ៉ាប់រងខ្នាតតូចក្នុងទម្រង់បែបណា?",('m',None,"ហត្ថលេខារបស់ក្រុមហ៊ុនធានារ៉ាប់រងខ្នាតតូច។")),
("guidance_05_chunk_3","តើវិញ្ញាបនបត្រធានារ៉ាប់រងខ្នាតតូច ត្រូវបញ្ជាក់និទ្ទេសអប្បបរមាអ្វីខ្លះ?",('h',9)),
("guidance_05_chunk_4","តើបណ្ណសន្យារ៉ាប់រងខ្នាតតូច ត្រូវមានមាតិកាអប្បបរមាអ្វីខ្លះ?",('h',2)),
("guidance_05_chunk_9","តើក្រុមហ៊ុនធានារ៉ាប់រងខ្នាតតូច ត្រូវផ្តល់ព័ត៌មានសង្ខេបអំពីការធានារ៉ាប់រងខ្នាតតូចជូនអតិថិជន តាមការលក់ប្រព័ន្ធទូរស័ព្ទ យ៉ាងណាខ្លះ?",('h',6)),
("guidance_05_chunk_10","តើកិច្ចសន្យាធានារ៉ាប់រងខ្នាតតូច ត្រូវបន្តសុពលភាពនៃការធានារ៉ាប់រងនៅពេលណា?",('m',None,"គិតចាប់ពីកាលបរិច្ឆេទទទួលបានការស្នើសុំ។")),
("guidance_05_chunk_12","តើក្រុមហ៊ុនធានារ៉ាប់រងខ្នាតតូច ត្រូវទូទាត់សំណងធានារ៉ាប់រងតាមមធ្យោបាយអ្វីខ្លះ?",('full',)),
("guidance_06_chunk_1","តើក្រុមហ៊ុនធានារ៉ាប់រង ត្រូវផ្ទេរហានិភ័យដែលលើសពីកម្រិតរក្សាទុកហានិភ័យទៅណា?",('full',)),
("guidance_06_chunk_2","តើក្រុមហ៊ុនធានារ៉ាប់រង ត្រូវប្រើប្រាស់ចំណាត់ថ្នាក់ឥណទានណា ក្នុងករណីក្រុមហ៊ុនធានារ៉ាប់រងបន្តក្រៅប្រទេសមានចំណាត់ថ្នាក់ឥណទានលើសពីមួយ?",('m',"ក្រុមហ៊ុនធានារ៉ាប់រង ត្រូវប្រើប្រាស់ចំណាត់ថ្នាក់ឥណទានដែលទាបជាងគេ","ដែលមានកម្រិតចំណាត់ថ្នាក់ផ្សេងគ្នា ។")),
("guidance_07_chunk_1","តើការក្លែងបន្លំធានារ៉ាប់រង សំដៅដល់អ្វី?",('full',)),
("guidance_07_chunk_4","តើក្រុមហ៊ុនធានារ៉ាប់រង ត្រូវរៀបចំគោលការណ៍ស្តីពីការគ្រប់គ្រងហានិភ័យនៃការក្លែងបន្លំធានារ៉ាប់រងតាមគោលការណ៍អប្បបរមាអ្វីខ្លះ?",('full',)),
("guidance_07_chunk_5","តើក្រុមហ៊ុនធានារ៉ាប់រង ត្រូវធ្វើដូចម្តេច ពេលរកឃើញសូចនាករនៃការក្លែងបន្លំធានារ៉ាប់រង?",('m',None,"ដើម្បីធានាបាននូវប្រសិទ្ធភាពនៃការអនុវត្ត។")),
("guidance_07_chunk_6","តើក្រុមហ៊ុនធានារ៉ាប់រង ត្រូវគ្រប់គ្រង និងកាត់បន្ថយហានិភ័យនៃការក្លែងបន្លំធានារ៉ាប់រងយ៉ាងណាខ្លះ?",('h',4)),
("guidance_07_chunk_7","តើក្រុមហ៊ុនធានារ៉ាប់រង ត្រូវរាយការណ៍អំពីការក្លែងបន្លំធានារ៉ាប់រងទៅកាន់នរណាខ្លះ?",('full',)),
("guidance_07_chunk_8","តើក្រុមហ៊ុនធានារ៉ាប់រង ត្រូវកត់ត្រាការក្លែងបន្លំធានារ៉ាប់រងដោយមានព័ត៌មានអប្បបរមាអ្វីខ្លះ?",('full',)),
("guidance_07_chunk_9","តើសវនករផ្ទៃក្នុងរបស់ក្រុមហ៊ុនធានារ៉ាប់រង ត្រូវធ្វើសវនកម្មលើអ្វី ទាក់ទងនឹងការក្លែងបន្លំធានារ៉ាប់រង?",('m',None,"ផ្តល់ការណែនាំជារៀងរាល់ឆ្នាំ។")),
("guidance_09_chunk_1","តើក្រុមហ៊ុនធានារ៉ាប់រង និងអង្គការផ្សាយពាណិជ្ជកម្មលើផលិតផលធានារ៉ាប់រង ត្រូវធ្វើអ្វីខ្លះ?",('full',)),
("guidance_09_chunk_2","តើក្រុមហ៊ុនធានារ៉ាប់រង និងអន្តរការីធានារ៉ាប់រង មិនត្រូវផ្សាយពាណិជ្ជកម្មដែលមានលក្ខណៈបែបណាខ្លះ?",('h',6)),
("guidance_09_chunk_3","តើអន្តរការីធានារ៉ាប់រងដែលមានបំណងផ្សាយពាណិជ្ជកម្មអំពីផលិតផលធានារ៉ាប់រង អាចធ្វើបែបណាខ្លះ?",('full',)),
("guidance_10_chunk_2","តើប្រភេទបច្ចេកវិទ្យាធានារ៉ាប់រងដែលអាចធ្វើការសាកល្បង ត្រូវមានលក្ខណៈវិនិច្ឆ័យអ្វីខ្លះ?",('full',)),
("guidance_10_chunk_3","តើបុគ្គលដែលមានបំណងសាកល្បងបច្ចេកវិទ្យាធានារ៉ាប់រង ត្រូវធ្វើដូចម្តេច ដើម្បីដាក់ពាក្យស្នើសុំការអនុញ្ញាត?",('full',)),
("guidance_10_chunk_4","តើបុគ្គលដែលអាចដាក់ពាក្យស្នើសុំសាកល្បងបច្ចេកវិទ្យាធានារ៉ាប់រង ត្រូវតែជានរណាខ្លះ?",('full',)),
("guidance_10_chunk_6","តើ ន.ធ.ក. (និយ័តករធានារ៉ាប់រងកម្ពុជា) មានរយៈពេលប៉ុន្មាន ក្នុងការសម្រេចផ្តល់លិខិតអនុញ្ញាត ឬបដិសេធលើពាក្យស្នើសុំ?",('full',)),
("guidance_10_chunk_7","តើក្នុងដំណាក់កាលសាកល្បង បុគ្គលបរិស្ថានសាកល្បង ត្រូវមានវិធានការការពារអតិថិជនអ្វីខ្លះ?",('hs',"ក- បុគ្គលបរិស្ថានសាកល្បងត្រូវមានវិធានការការពារអតិថិជន",5)),
("guidance_10_chunk_11","តើបុគ្គលបរិស្ថានសាកល្បង ត្រូវភ្ជាប់របាយការណ៍អ្វីខ្លះ ពេលស្នើសុំចាកចេញពីបរិស្ថានសាកល្បងបច្ចេកវិទ្យាធានារ៉ាប់រង?",('h',4)),
("guidance_10_chunk_13","តើក្រុមហ៊ុនធានារ៉ាប់រង ឬក្រុមហ៊ុនបច្ចេកវិទ្យា ត្រូវបង់កម្រៃសេវារដ្ឋបាលអ្វីខ្លះ ទាក់ទងនឹងបរិស្ថានសាកល្បងបច្ចេកវិទ្យាធានារ៉ាប់រង?",('h',4)),
("guidance_10_chunk_14","តើ ន.ធ.ក. (និយ័តករធានារ៉ាប់រងកម្ពុជា) នឹងពិចារណាបន្ធូរបន្ថយ ឬលើកលែងលក្ខខណ្ឌតម្រូវបទប្បញ្ញត្តិយ៉ាងណា ដើម្បីជំរុញនវានុវត្តន៍?",('m',None,"ដោយអាស្រ័យលើប្រព័ន្ធបច្ចេកវិទ្យាធានារ៉ាប់រងដែលជាកម្មវត្ថុនៃការសាកល្បង។")),
("guidance_10_chunk_15","តើវិធានការផ្សេងទៀតអាចត្រូវបានអនុវត្តចំពោះបុគ្គលបរិស្ថានសាកល្បង ក្នុងករណីបានធ្វើសកម្មភាពអ្វីខ្លះ?",('full',)),
("guidance_11_chunk_2","តើភ្នាក់ងារធានារ៉ាប់រង ត្រូវធ្វើអ្វីខ្លះ ដើម្បីមានសមត្ថភាពវិជ្ជាជីវៈក្នុងការផ្តល់សេវាកម្មធានារ៉ាប់រង?",('full',)),
("guidance_11_chunk_3","តើភ្នាក់ងារធានារ៉ាប់រង ត្រូវប្រកាន់យកនូវអាកប្បកិរិយាអ្វីខ្លះ ក្នុងផ្នែកសុចរិតភាព និងភាពស្មោះត្រង់?",('h',5)),
("guidance_11_chunk_4","តើភ្នាក់ងារធានារ៉ាប់រង ត្រូវប្រកាន់យកនូវអាកប្បកិរិយាអ្វីខ្លះ ក្នុងការទទួលខុសត្រូវប្រកបដោយវិជ្ជាជីវៈ?",('h',3)),
("guidance_11_chunk_5","តើភ្នាក់ងារធានារ៉ាប់រង ត្រូវប្រកាន់យកនូវឥរិយាបថអ្វីខ្លះ ក្នុងការប្រតិបត្តិវិជ្ជាជីវៈ?",('h',4)),
("guidance_11_chunk_6","តើភ្នាក់ងារធានារ៉ាប់រង ត្រូវប្រកាន់យកនូវអាកប្បកិរិយាអ្វីខ្លះ ដើម្បីរក្សាការសម្ងាត់វិជ្ជាជីវៈ?",('full',)),
("guidance_11_chunk_7","តើភ្នាក់ងារធានារ៉ាប់រង ត្រូវប្រកាន់យកនូវឥរិយាបថអ្វីខ្លះ ដើម្បីជៀសវាងទំនាស់ផលប្រយោជន៍?",('h',5)),
("guidance_11_chunk_8","តើភ្នាក់ងារធានារ៉ាប់រង ត្រូវប្រកាន់យកនូវឥរិយាបថអ្វីខ្លះ ក្នុងការបន្តបម្រើអតិថិជនបន្ទាប់ពីការចេញបណ្ណសន្យារ៉ាប់រង?",('full',)),
("guidance_11_chunk_9","តើក្រុមហ៊ុនធានារ៉ាប់រង ត្រូវធ្វើអ្វី ចំពោះភ្នាក់ងារធានារ៉ាប់រងដែលបំពានក្រមសីលធម៌?",('full',)),
("guidance_11_chunk_10","តើការបំពានក្រមសីលធម៌កម្រិតធ្ងន់ធ្ងររបស់ភ្នាក់ងារធានារ៉ាប់រង រួមមានអ្វីខ្លះ?",('hs',"ការបំពានកម្រិតធ្ងន់ធ្ងរ ត្រូវបានកំណត់ដូចខាងក្រោម៖",3)),
("guidance_11_chunk_11","តើភ្នាក់ងារធានារ៉ាប់រងដែលបំពានក្រមសីលធម៌កម្រិតស្រាល ត្រូវទទួលរងវិធានការអ្វីខ្លះ?",('h',6)),
("guidance_11_chunk_12","តើក្រុមហ៊ុនធានារ៉ាប់រង ត្រូវផ្តល់វគ្គបណ្តុះបណ្តាលអំពីក្រមសីលធម៌វិជ្ជាជីវៈភ្នាក់ងារធានារ៉ាប់រងជូនភ្នាក់ងារនៅពេលណា?",('full',)),
("guidance_12_chunk_2","តើក្រុមហ៊ុនធានារ៉ាប់រង ត្រូវផ្តល់ពាក្យស្នើសុំទាមទារសំណងធានារ៉ាប់រងជូនអ្នកទាមទារសំណងក្នុងរយៈពេលប៉ុន្មាន?",('m',None,"ហានិភ័យធានារ៉ាប់រង។")),
("guidance_12_chunk_3","តើក្រុមហ៊ុនធានារ៉ាប់រង ត្រូវឆ្លើយតបអំពីការបដិសេធ ឬការសម្រេចផ្តល់សំណងធានារ៉ាប់រងក្នុងរយៈពេលប៉ុន្មានថ្ងៃ?",('m',None,"និងឯកសារតម្រូវគ្រប់គ្រាន់។")),
("guidance_12_chunk_4","តើក្រុមហ៊ុនធានារ៉ាប់រង ត្រូវទូទាត់សំណងធានារ៉ាប់រងជូនអ្នកទាមទារសំណងតាមការកំណត់បែបណាខ្លះ?",('full',)),
("guidance_12_chunk_5","តើក្រុមហ៊ុនធានារ៉ាប់រង ត្រូវកត់ត្រាសំណងធានារ៉ាប់រងដោយមានព័ត៌មានអប្បបរមាអ្វីខ្លះ?",('h',9)),
("guidance_12_chunk_6","តើក្រុមហ៊ុនធានារ៉ាប់រង ត្រូវរក្សាទុកឯកសារអំពីការទទួល និងដោះស្រាយសំណងធានារ៉ាប់រងក្នុងរយៈពេលអប្បបរមាប៉ុន្មានឆ្នាំ?",('full',)),
("guidance_12_chunk_8","តើក្រុមហ៊ុនធានារ៉ាប់រង ត្រូវផ្តល់របាយការណ៍អំពីការទទួល និងដោះស្រាយសំណងធានារ៉ាប់រងមក ន.ធ.ក. (និយ័តករធានារ៉ាប់រងកម្ពុជា) នៅពេលណាខ្លះ?",('full',)),
("guidance_12_chunk_9","តើក្រុមហ៊ុនធានារ៉ាប់រង ត្រូវបង់កម្រៃសេវារដ្ឋបាលអ្វីខ្លះ ទាក់ទងនឹងគោលការណ៍ស្តីពីការទទួល និងដោះស្រាយសំណងធានារ៉ាប់រង?",('m',None,"៥០០ ០០០ (ប្រាំរយពាន់) រៀល។")),
("guidance_13_chunk_1","តើក្រុមហ៊ុនជើងសាធានារ៉ាប់រង និងក្រុមហ៊ុនភ្នាក់ងារធានារ៉ាប់រង ត្រូវរៀបចំរបាយការណ៍អ្វីខ្លះ?",('full',)),
("guidance_13_chunk_2","តើក្រុមហ៊ុនជើងសាធានារ៉ាប់រង ត្រូវផ្តល់របាយការណ៍ប្រចាំខែ ប្រចាំត្រីមាស និងប្រចាំឆ្នាំ ត្រឹមពេលណាខ្លះ?",('hs',"ក្រុមហ៊ុនជើងសាធានារ៉ាប់រង ត្រូវផ្តល់មក",3)),
("guidance_15_chunk_1","តើបុគ្គលដែលមានបំណងចូលបម្រើការងារផ្តល់សេវាលក់ផលិតផលធានារ៉ាប់រង ត្រូវធ្វើដូចម្តេច?",('full',)),
("guidance_15_chunk_3","តើសមាគមធានារ៉ាប់រងកម្ពុជា ត្រូវរៀបចំអ្វីខ្លះ ទាក់ទងនឹងឯកសារ និងវិធីសាស្ត្របង្រៀន?",('full',)),
("guidance_15_chunk_4","តើសិក្ខាកាមដែលចាត់ទុកថាប្រឡងជាប់វគ្គបណ្តុះបណ្តាលមូលដ្ឋាន ត្រូវមានកម្រិតពិន្ទុចាប់ពីប៉ុន្មានភាគរយ?",('m',None,"ឡើងទៅ។")),
("guidance_15_chunk_5","តើនៅក្នុងដំណាក់កាលអន្តរកាល អ្នកណាត្រូវទទួលខុសត្រូវរៀបចំផ្តល់វគ្គបណ្តុះបណ្តាលមូលដ្ឋាន?",('full',)),
("guidance_15_chunk_6","តើក្រុមហ៊ុនធានារ៉ាប់រងដែលមានបំណងផ្តល់វគ្គបណ្តុះបណ្តាល ត្រូវរៀបចំអ្វីខ្លះ?",('full',)),
("guidance_15_chunk_7","តើបុគ្គលដែលមានបំណងប្រកបអាជីពជាភ្នាក់ងារធានារ៉ាប់រង ត្រូវធ្វើដូចម្តេច បន្ទាប់ពីបានវិញ្ញាបនបត្របញ្ចប់វគ្គបណ្តុះបណ្តាលមូលដ្ឋាន?",('m',None,"ឬម៉ូដែលភ្នាក់ងារលក់ផ្ទាល់។")),
("guidance_16_chunk_2","តើបណ្ណសន្យារ៉ាប់រង ត្រូវមានមាតិកាអប្បបរមាអ្វីខ្លះ?",('hs',"បណ្ណសន្យារ៉ាប់រងត្រូវរៀបចំឡើងក្នុងទម្រង់",4)),
("guidance_16_chunk_2","តើពេលបញ្ចប់កិច្ចសន្យាធានារ៉ាប់រងមុនកាលកំណត់ ម្ចាស់បណ្ណសន្យារ៉ាប់រងមានសិទ្ធិទទួលបានតម្លៃប៉ុន្មាន?",('m',"ការបញ្ចប់កិច្ចសន្យាធានារ៉ាប់រង៖ កំណត់លក្ខខណ្ឌ","ដែលត្រូវកំណត់នៅក្នុងវិញ្ញាបនបត្រធានារ៉ាប់រង")),
("guidance_16_chunk_4","តើកំណត់សំអាង គឺជាអ្វី និងត្រូវបញ្ជាក់ខ្លឹមសារអប្បបរមាអ្វីខ្លះ?",('full',)),
],
# To do: author these the same way (read the chunk file, slice verbatim passages).
"laws": [],
"sub_decrees": [],
"prakas": [],
}


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in CONFIG:
        sys.exit("usage: python build_gold_questions.py <guidance|laws|sub_decrees|prakas>")
    build(sys.argv[1])
