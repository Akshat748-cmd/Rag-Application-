"""
Text Chunking Module
Supports: Fixed Size, Recursive, Sentence-based, Paragraph, Token, Sliding Window, Semantic chunking
"""
import re
from typing import List, Dict

# Pre-compiled sentence boundary pattern (compiled once at import time)
_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+')

# Sentence-ending punctuation for boundary detection
_SENTENCE_END = {'.', '!', '?'}


def _find_sentence_boundary(words: list, ideal_end: int, search_window: int = 50) -> int:
    """
    Given a list of words and an ideal end index, search backward (up to search_window words)
    to find the last word that ends with sentence-ending punctuation.
    Returns the adjusted end index (after the sentence-ending word).
    Falls back to ideal_end if no boundary found.
    """
    search_start = max(ideal_end - search_window, 0)
    for i in range(ideal_end - 1, search_start - 1, -1):
        word = words[i].rstrip('"\')')  # strip trailing quotes/brackets
        if word and word[-1] in _SENTENCE_END:
            return i + 1  # cut AFTER this word
    return ideal_end  # no boundary found, use original cut point


def fixed_size_chunk(text: str, chunk_size: int = 200, overlap: int = 20) -> List[Dict]:
    """Split text into fixed-size chunks with optional overlap.
    Chunks always end at a complete sentence boundary."""
    words = text.split()
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(words):
        ideal_end = min(start + chunk_size, len(words))

        # Snap to nearest sentence boundary (only if not at end of text)
        if ideal_end < len(words):
            end = _find_sentence_boundary(words, ideal_end)
        else:
            end = ideal_end

        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        chunks.append({
            "id": chunk_id,
            "text": chunk_text,
            "word_count": len(chunk_words),
            "start_word": start,
            "end_word": end,
            "strategy": "fixed"
        })
        chunk_id += 1

        if end >= len(words):
            break
        start = max(end - overlap, start + 1)  # ensure forward progress

    return chunks


def recursive_chunk(text: str, max_size: int = 200, overlap: int = 20) -> List[Dict]:
    """Recursively split text using paragraph → sentence → word boundaries."""
    chunks = []
    chunk_id = 0

    def split_recursive(txt: str, separators: list):
        if not separators:
            return [txt]
        sep = separators[0]
        parts = txt.split(sep)
        results = []
        current = ""
        current_word_count = 0
        for part in parts:
            part_words = len(part.split()) if part else 0
            candidate_count = current_word_count + (1 if current else 0) + part_words
            if candidate_count <= max_size:
                current = current + sep + part if current else part
                current_word_count = candidate_count
            else:
                if current:
                    results.append(current.strip())
                current = part
                current_word_count = part_words
        if current:
            results.append(current.strip())
        return results

    raw_chunks = split_recursive(text, ["\n\n", "\n", ". ", " "])

    for i, chunk_text in enumerate(raw_chunks):
        if chunk_text.strip():
            chunks.append({
                "id": chunk_id,
                "text": chunk_text.strip(),
                "word_count": len(chunk_text.split()),
                "start_word": i,
                "end_word": i + 1,
                "strategy": "recursive"
            })
            chunk_id += 1

    return chunks


def sentence_chunk(text: str, sentences_per_chunk: int = 3, overlap: int = 1) -> List[Dict]:
    """Split text into chunks of N sentences.

    overlap is in SENTENCES (not words). It is safely clamped so that
    it never equals or exceeds sentences_per_chunk (which would cause
    the window to stop advancing or go backwards).
    """
    # Use pre-compiled pattern; filter empty strings in one pass
    sentences = [s for s in _SENTENCE_RE.split(text.strip()) if s.strip()]

    if not sentences:
        return []

    # sentences_per_chunk must be at least 1
    sentences_per_chunk = max(1, sentences_per_chunk)

    # overlap in sentences: clamp to [0, sentences_per_chunk - 1]
    # This ensures the window always moves forward by at least 1 sentence
    safe_overlap = max(0, min(overlap, sentences_per_chunk - 1))

    chunks = []
    chunk_id = 0
    start = 0

    while start < len(sentences):
        end = min(start + sentences_per_chunk, len(sentences))
        chunk_sentences = sentences[start:end]
        chunk_text = " ".join(chunk_sentences)
        chunks.append({
            "id": chunk_id,
            "text": chunk_text,
            "word_count": len(chunk_text.split()),
            "sentence_count": len(chunk_sentences),
            "start_sentence": start,
            "end_sentence": end,
            "strategy": "sentence"
        })
        chunk_id += 1

        if end >= len(sentences):
            break
        # Move forward: next chunk starts (sentences_per_chunk - safe_overlap) after current start
        step = sentences_per_chunk - safe_overlap
        start += max(step, 1)  # guarantee forward progress

    return chunks


def paragraph_chunk(text: str, max_size: int = 200, overlap: int = 0) -> List[Dict]:
    """Split text by paragraph boundaries (blank lines). Merges small paras into one chunk.
    If a single paragraph exceeds max_size, it is split by sentences to avoid huge chunks.
    """
    raw_paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

    # If no paragraph breaks found, treat each line as a paragraph
    if len(raw_paras) <= 1:
        raw_paras = [p.strip() for p in text.split('\n') if p.strip()]

    chunks = []
    chunk_id = 0
    buffer = []
    buffer_words = 0

    max_size = max(10, max_size)  # sanity floor

    def flush_buffer():
        nonlocal chunk_id, buffer, buffer_words
        if buffer:
            chunk_text = "\n\n".join(buffer)
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "word_count": len(chunk_text.split()),
                "paragraph_count": len(buffer),
                "strategy": "paragraph"
            })
            chunk_id += 1
            buffer = []
            buffer_words = 0

    for para in raw_paras:
        para_words = len(para.split())

        # If single paragraph is larger than max_size, split it by sentences first
        if para_words > max_size:
            flush_buffer()
            # Split oversized paragraph by sentence
            sub_sentences = [s.strip() for s in _SENTENCE_RE.split(para) if s.strip()]
            sub_buf = []
            sub_words = 0
            for sent in sub_sentences:
                sw = len(sent.split())
                if sub_words + sw <= max_size:
                    sub_buf.append(sent)
                    sub_words += sw
                else:
                    if sub_buf:
                        ct = " ".join(sub_buf)
                        chunks.append({
                            "id": chunk_id,
                            "text": ct,
                            "word_count": len(ct.split()),
                            "paragraph_count": 1,
                            "strategy": "paragraph"
                        })
                        chunk_id += 1
                    sub_buf = [sent]
                    sub_words = sw
            if sub_buf:
                ct = " ".join(sub_buf)
                chunks.append({
                    "id": chunk_id,
                    "text": ct,
                    "word_count": len(ct.split()),
                    "paragraph_count": 1,
                    "strategy": "paragraph"
                })
                chunk_id += 1
            continue

        if buffer_words + para_words <= max_size:
            buffer.append(para)
            buffer_words += para_words
        else:
            flush_buffer()
            buffer = [para]
            buffer_words = para_words

    flush_buffer()
    return chunks


def token_chunk(text: str, chunk_size: int = 512, overlap: int = 50) -> List[Dict]:
    """Split by approximate token count (1 token ≈ 4 chars). Ideal for LLM context windows."""
    tokens = re.findall(r'\S+|\s+', text)
    token_chars = [len(t) for t in tokens]
    char_limit = chunk_size * 4

    chunks = []
    chunk_id = 0
    start = 0

    while start < len(tokens):
        end = start
        char_count = 0
        while end < len(tokens) and char_count + token_chars[end] <= char_limit:
            char_count += token_chars[end]
            end += 1
        if end == start:
            end = start + 1

        chunk_text = "".join(tokens[start:end]).strip()
        if chunk_text:
            approx_tokens = sum(token_chars[start:end]) // 4
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "word_count": len(chunk_text.split()),
                "approx_tokens": approx_tokens,
                "strategy": "token"
            })
            chunk_id += 1

        overlap_chars = overlap * 4
        back = 0
        idx = end - 1
        while idx >= start and back < overlap_chars:
            back += token_chars[idx]
            idx -= 1
        start = max(idx + 1, start + 1)

    return chunks


def sliding_window_chunk(text: str, chunk_size: int = 100, overlap: int = 50) -> List[Dict]:
    """Dense sliding window with sentence-aware boundaries.

    FIX 1: overlap is capped at 50% of chunk_size so step is always >= chunk_size//2.
    FIX 2: Each chunk ends at a complete sentence boundary — no half-sentences.
    """
    words = text.split()
    chunks = []
    chunk_id = 0

    # FIX 1: Cap overlap at 50% of chunk_size — ensures meaningful forward movement
    # Example: chunk_size=20, overlap=20 → safe_overlap=10, step=10 (was step=1 before!)
    safe_overlap = min(overlap, chunk_size // 2)
    step = max(chunk_size - safe_overlap, 1)

    start = 0

    while start < len(words):
        ideal_end = min(start + chunk_size, len(words))

        # FIX 2: Snap to sentence boundary (only when not at end of text)
        if ideal_end < len(words):
            end = _find_sentence_boundary(words, ideal_end)
        else:
            end = ideal_end

        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        chunks.append({
            "id": chunk_id,
            "text": chunk_text,
            "word_count": len(chunk_words),
            "start_word": start,
            "end_word": end,
            "strategy": "sliding_window"
        })
        chunk_id += 1

        if end >= len(words):
            break
        start += step

    return chunks

def semantic_chunk(
    text: str,
    similarity_threshold: float = 0.75,
    min_chunk_size: int = 30,
    max_chunk_size: int = 300
) -> List[Dict]:
    """Group sentences into chunks by embedding similarity.

    Algorithm:
      1. Split text into sentences using the pre-compiled _SENTENCE_RE.
      2. Embed each sentence with 'all-MiniLM-L6-v2' (lazy-loaded).
      3. Walk consecutive sentence pairs; start a new chunk when the cosine
         similarity between adjacent embeddings drops below `similarity_threshold`,
         or when the current buffer exceeds `max_chunk_size` words.
      4. Merge any tiny trailing chunk (< min_chunk_size words) into the previous one.

    Args:
        text: Input text to chunk.
        similarity_threshold: Cosine similarity below which a new chunk starts (0–1).
        min_chunk_size: Minimum words per chunk; tiny chunks are merged with the previous.
        max_chunk_size: Maximum words per chunk before a forced split.

    Returns:
        List of chunk dicts compatible with all other strategies.
    """
    # Lazy import — only needed for this strategy
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "semantic chunking requires 'sentence-transformers' and 'numpy'. "
            "Run: pip install sentence-transformers numpy"
        )

    sentences = [s.strip() for s in _SENTENCE_RE.split(text.strip()) if s.strip()]
    if not sentences:
        return []

    if len(sentences) == 1:
        return [{
            "id": 0,
            "text": sentences[0],
            "word_count": len(sentences[0].split()),
            "sentence_count": 1,
            "strategy": "semantic"
        }]

    # Embed all sentences at once (batch is more efficient)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sentences, show_progress_bar=False, convert_to_numpy=True)

    # Normalise for cosine similarity via dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)  # avoid div-by-zero
    embeddings_norm = embeddings / norms

    # Group sentences into semantic chunks
    groups: List[List[str]] = []
    current_group: List[str] = [sentences[0]]
    current_words = len(sentences[0].split())

    for i in range(1, len(sentences)):
        sim = float(np.dot(embeddings_norm[i - 1], embeddings_norm[i]))
        sent_words = len(sentences[i].split())

        # Start new chunk on similarity drop OR max_size overflow
        if sim < similarity_threshold or (current_words + sent_words) > max_chunk_size:
            groups.append(current_group)
            current_group = [sentences[i]]
            current_words = sent_words
        else:
            current_group.append(sentences[i])
            current_words += sent_words

    if current_group:
        groups.append(current_group)

    # Merge tiny trailing groups into the previous chunk
    merged: List[List[str]] = []
    for grp in groups:
        grp_words = sum(len(s.split()) for s in grp)
        if merged and grp_words < min_chunk_size:
            merged[-1].extend(grp)
        else:
            merged.append(grp)

    # Build output dicts
    chunks = []
    for chunk_id, grp in enumerate(merged):
        chunk_text = " ".join(grp)
        chunks.append({
            "id": chunk_id,
            "text": chunk_text,
            "word_count": len(chunk_text.split()),
            "sentence_count": len(grp),
            "strategy": "semantic"
        })

    return chunks


def chunk_text(text: str, strategy: str = "fixed", **kwargs) -> List[Dict]:
    """Main chunking function — dispatches to the right strategy."""
    strategies = {
        "fixed":          fixed_size_chunk,
        "recursive":      recursive_chunk,
        "sentence":       sentence_chunk,
        "paragraph":      paragraph_chunk,
        "token":          token_chunk,
        "sliding_window": sliding_window_chunk,
        "semantic":       semantic_chunk,
    }
    if strategy not in strategies:
        strategy = "fixed"

    return strategies[strategy](text, **kwargs)
