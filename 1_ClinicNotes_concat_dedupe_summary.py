import re
import math
import time
import argparse
import pandas as pd
from tqdm import tqdm
from llama_cpp import Llama
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

# ========== Config ==========
REPO_ID = "mradermacher/Qwen2.5-7B-Instruct-medical_summary_latest-GGUF"
FILENAME = "Qwen2.5-7B-Instruct-medical_summary_latest.IQ4_XS.gguf"

TEXT_COL = "CLINICAL_DOCUMENT_TEXT"  ## this is the original clinical notes col
DROP_FUTURE_COL = "CLINICAL_NOTE_TEXT_DROP_FUTURE"
INPUT_CSV = "input file path"
OUTPUT_CSV = "output file path"


# Model / context settings
# Match the Qwen2.5 model's trained context length. The prior 128k allocation
# consumed excessive KV/compute memory on a 48 GB GPU before inference began.
MAX_CONTEXT_TOKENS = 32768
RESERVED_PROMPT_TOKENS = 1200
RESERVED_OUTPUT_TOKENS = 220
CHUNK_OVERLAP_TOKENS = 64
TEMPERATURE = 0.0
TOP_P = 1.0

# ========== Load llama.cpp model ONCE, lazily ==========
llm = None
MODEL_PATH = None


def get_llm():
    global llm
    if llm is not None:
        return llm
    if MODEL_PATH:
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=MAX_CONTEXT_TOKENS,
            n_gpu_layers=-1,
            verbose=True,
        )
    else:
        llm = Llama.from_pretrained(
            repo_id=REPO_ID,
            filename=FILENAME,
            n_ctx=MAX_CONTEXT_TOKENS,
            n_gpu_layers=-1,
            verbose=True,
        )
    return llm

# ========== Helpers to keep complete sentences / normalize ==========
_ABBR_SET = {"mr", "mrs", "ms", "dr", "prof", "st", "vs", "no", "fig", "ref", "e.g", "i.e"}
_MULTIWORD_ABBR = {"et al"}
_SECTIONS_TO_DROP = [
    "ASSESSMENT / PLAN",
    "OUTPATIENT FOLLOWUP",
    "DISCHARGE DISPOSITION",
    "ADVANCE DIRECTIVES",
]


def _ends_with_abbrev(prefix: str) -> bool:
    tail = re.sub(r"\s+", " ", prefix[-30:].strip())
    tokens = tail.split(" ")
    if not tokens:
        return False
    last = tokens[-1].rstrip(".").lower()
    if last in _ABBR_SET:
        return True
    if len(tokens) >= 2:
        last2 = (tokens[-2] + " " + tokens[-1].rstrip(".")).lower()
        if last2 in _MULTIWORD_ABBR:
            return True
    return False


def clip_to_last_full_sentence(text: str) -> str:
    if not isinstance(text, str):
        return text
    t = re.sub(r"\s+", " ", text).strip()
    last_end = -1
    for i, ch in enumerate(t):
        if ch in ".!?":
            if not _ends_with_abbrev(t[:i]):
                last_end = i
    if last_end != -1:
        return t[: last_end + 1].strip()
    t = re.sub(r"\W*\w?$", "", t).strip()
    return (t + ".") if t else t


def normalize_model_output(text: str) -> str:
    if not isinstance(text, str):
        return ""
    full_to_half = str.maketrans(
        {
            "０": "0",
            "１": "1",
            "２": "2",
            "３": "3",
            "４": "4",
            "５": "5",
            "６": "6",
            "７": "7",
            "８": "8",
            "９": "9",
        }
    )
    text = text.translate(full_to_half)
    text = re.sub(r"(\d)\s+", r"\1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_into_sentences(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    text = normalize_model_output(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [clip_to_last_full_sentence(part).strip() for part in parts if part and part.strip()]


def sentence_signature(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def concat_and_dedup_chunk_summaries(chunk_summaries: list[str]) -> str:
    unique_sentences = []
    seen = set()

    for summary in chunk_summaries:
        for sentence in split_into_sentences(summary):
            signature = sentence_signature(sentence)
            if not signature or signature in seen:
                continue
            seen.add(signature)
            unique_sentences.append(sentence)

    combined = " ".join(unique_sentences).strip()
    combined = normalize_model_output(combined)
    combined = clip_to_last_full_sentence(combined)
    return combined


def remove_future_event_sections(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""

    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    target_heading_pattern = "|".join(re.escape(section) for section in _SECTIONS_TO_DROP)
    target_heading_re = re.compile(rf"(?i)\b({target_heading_pattern})(?::)?\b")
    next_heading_re = re.compile(
        r"(?:(?<=^)|(?<=\n)|(?<=[.!?]\s)|(?<=  ))"
        r"([A-Z][A-Z0-9/&,\-]*(?: [A-Z0-9/&,\-]+){0,8})(?::)?(?=(?:\n|  |$))"
    )

    while True:
        match = target_heading_re.search(cleaned)
        if not match:
            break

        next_match = next_heading_re.search(cleaned, match.end())
        end_idx = next_match.start() if next_match else len(cleaned)
        cleaned = (cleaned[:match.start()] + " " + cleaned[end_idx:]).strip()

    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r" *\n *", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r" {2,}", " ", cleaned).strip()
    return cleaned


# ========== Token utilities ==========
def _tok(text: str) -> list[int]:
    return get_llm().tokenize(text.encode("utf-8", errors="ignore"))


def _detok(tokens: list[int]) -> str:
    return get_llm().detokenize(tokens).decode("utf-8", errors="ignore")


def truncate_text_by_tokens(text: str, max_tokens: int) -> str:
    ids = _tok(text)
    if len(ids) <= max_tokens:
        return text
    return _detok(ids[:max_tokens])


def chunk_text_by_tokens(
    text: str,
    max_ctx_tokens: int = MAX_CONTEXT_TOKENS,
    reserved_prompt_tokens: int = RESERVED_PROMPT_TOKENS,
    reserved_output_tokens: int = RESERVED_OUTPUT_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """
    Split a long text into token-sized chunks so that:
      len(system+user prompt) + len(chunk) + reserved_output_tokens <= max_ctx_tokens
    We conservatively reserve `reserved_prompt_tokens` for chat template overhead.
    """
    if not text or not str(text).strip():
        return []

    max_user_tokens = max_ctx_tokens - reserved_prompt_tokens - reserved_output_tokens
    if max_user_tokens <= 512:
        max_user_tokens = max_ctx_tokens // 2

    ids = _tok(str(text))
    n = len(ids)
    if n <= max_user_tokens:
        return [text]

    chunks = []
    start = 0
    while start < n:
        end = min(start + max_user_tokens, n)
        chunk_ids = ids[start:end]
        chunks.append(_detok(chunk_ids))
        if end == n:
            break
        start = end - overlap_tokens
    return chunks


# ========== LLM calls with retries ==========
_SYSTEM_PROMPT = (
    """You are a medical assistant.
        Summarize the following clinical note in 3–4 concise sentences.
        Include only the following elements (if present and relevant to oncology and/or hematology):
        - Pathology findings
        - Radiology findings
        - Ancillary studies and relevant past medical history
        If no oncology or hematology–related information is present, output: "No relevant oncology or hematology information."

        Output requirements:
        - Use only English.
        - Do not include bullet points, lists, or headings.
        - Do not add interpretations or information not present in the note.
        - Keep the summary factual and concise.
"""
)


def _safe_chat_completion(
    messages,
    max_tokens=RESERVED_OUTPUT_TOKENS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    max_retries=3,
):
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = get_llm().create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            return resp["choices"][0]["message"]["content"].strip()
        except ValueError as e:
            last_err = e
            if "exceed context window" in str(e) or "Requested tokens" in str(e):
                raise
            time.sleep(0.5 * attempt)
        except Exception as e:
            last_err = e
            time.sleep(0.5 * attempt)
    raise last_err


def summarize_chunk(chunk_text: str) -> str:
    user_prompt = f"Summarize the following note in English only:\n\n{chunk_text.strip()}"
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        return _safe_chat_completion(messages)
    except ValueError:
        shrunk = chunk_text
        for _ in range(3):
            ids = _tok(shrunk)
            if len(ids) < 256:
                break
            new_len = int(math.floor(len(ids) * 0.8))
            shrunk = _detok(ids[:new_len])
            messages[1]["content"] = f"Summarize the following note in English only:\n\n{shrunk.strip()}"
            try:
                return _safe_chat_completion(messages)
            except ValueError:
                continue
        return ""


def summarize_text(text: str) -> str:
    """
    High-level pipeline:
      1) chunk the note by tokens
      2) summarize each chunk
      3) concatenate unique chunk-summary sentences without a second summarization pass
    """
    if pd.isna(text) or not str(text).strip():
        return ""

    text = remove_future_event_sections(str(text))
    if not text.strip():
        return ""

    chunks = chunk_text_by_tokens(
        text,
        max_ctx_tokens=MAX_CONTEXT_TOKENS,
        reserved_prompt_tokens=RESERVED_PROMPT_TOKENS,
        reserved_output_tokens=RESERVED_OUTPUT_TOKENS,
        overlap_tokens=CHUNK_OVERLAP_TOKENS,
    )

    chunk_summaries = []
    for ch in chunks:
        summary = summarize_chunk(ch)
        summary = normalize_model_output(summary)
        summary = clip_to_last_full_sentence(summary)
        if summary:
            chunk_summaries.append(summary)

    if not chunk_summaries:
        return ""

    return concat_and_dedup_chunk_summaries(chunk_summaries)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate oncology-focused summaries for clinical-note rows.")
    parser.add_argument("--input", default=INPUT_CSV)
    parser.add_argument("--output", default=OUTPUT_CSV)
    parser.add_argument("--model-path", default=None, help="Optional local GGUF path; otherwise use the configured Hugging Face repo.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    global MODEL_PATH
    args = parse_args()
    MODEL_PATH = args.model_path
    df = pd.read_csv(args.input, low_memory=False)

    print(f"New dataset len: {len(df)}")

    def _safe_row_summarize(text):
        try:
            return summarize_text(text)
        except Exception as e:
            return f"[error] {type(e).__name__}: {str(e)[:200]}"

    if args.overwrite and os.path.isfile(args.output):
        os.remove(args.output)
    output_exists = os.path.isfile(args.output)
    completed_keys = set()
    if args.resume and output_exists:
        existing = pd.read_csv(args.output, dtype=str, low_memory=False)
        if "CLINICAL_DOCUMENT_FPK" in existing.columns:
            completed_keys = set(existing["CLINICAL_DOCUMENT_FPK"].dropna().astype(str))
            print(f"Resume: skipping {len(completed_keys):,} completed document(s).")

    print(f"Starting processing. Saving incrementally to: {args.output}")
    pending = df.loc[~df["CLINICAL_DOCUMENT_FPK"].astype(str).isin(completed_keys)]
    if not pending.empty:
        # Initialize once up front so model/context failures stop the job instead
        # of being converted into one error row per document.
        get_llm()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Summarizing"):
        document_key = str(row.get("CLINICAL_DOCUMENT_FPK", ""))
        if document_key and document_key in completed_keys:
            continue
        cleaned_text = remove_future_event_sections(row[TEXT_COL])
        summary_result = _safe_row_summarize(cleaned_text)

        current_row_df = pd.DataFrame([row])
        current_row_df[DROP_FUTURE_COL] = cleaned_text
        current_row_df["summary"] = summary_result

        current_row_df.to_csv(
            args.output,
            mode="a",
            header=not output_exists,
            index=False,
            encoding="utf-8-sig",
        )

        output_exists = True

    print(f"✅ Finished! All rows saved to: {args.output}")


if __name__ == "__main__":
    main()
