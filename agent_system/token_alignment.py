"""Verified character offsets for the original generated response tokens."""

import json
from bisect import bisect_left, bisect_right
from functools import lru_cache


# GPT-2's reversible byte alphabet, also used by Llama 3 and Qwen ByteLevel
# decoders. Bytes that are not printable Latin-1 use code points above 255.
_PRINTABLE_BYTES = [*range(33, 127), *range(161, 173), *range(174, 256)]
_OTHER_BYTES = [value for value in range(256) if value not in _PRINTABLE_BYTES]
_BYTE_VALUES = {chr(value): value for value in _PRINTABLE_BYTES}
_BYTE_VALUES.update({chr(256 + index): value for index, value in enumerate(_OTHER_BYTES)})


@lru_cache(maxsize=32768)
def _token_bytes(token):
    return bytes(_BYTE_VALUES[char] for char in token)


def _byte_level_offsets(tokenizer, token_ids, output):
    """Map byte intervals to Unicode character intervals, including split bytes."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    decoder = getattr(backend, "decoder", None)
    if decoder is None or json.loads(decoder.__getstate__()).get("type") != "ByteLevel":
        return None
    try:
        pieces = [_token_bytes(token) for token in tokenizer.convert_ids_to_tokens(token_ids)]
        text = b"".join(pieces).decode("utf-8")
    except (KeyError, UnicodeDecodeError):
        # Added tokens outside the byte alphabet or malformed UTF-8 require
        # the conservative, verified tokenizer-offset fallback below.
        return None
    if text != output:
        return None
    boundaries = [0]
    for char in text:
        boundaries.append(boundaries[-1] + len(char.encode("utf-8")))
    offsets = []
    cursor = 0
    for piece in pieces:
        end = cursor + len(piece)
        # Every token contributing bytes to a character overlaps that character.
        offsets.append((bisect_right(boundaries, cursor) - 1, bisect_left(boundaries, end)))
        cursor = end
    return offsets


def response_token_offsets(tokenizer, response_ids, output):
    """Return (original positions, character offsets, alignment status).

    ByteLevel tokenizers use the generated token bytes directly, so alternative
    tokenizations of the same text remain alignable. Other decoders retain the
    exact-ID re-encoding check before trusting their offsets. Callers must decode
    ``output`` with skip_special_tokens=True, clean_up_tokenization_spaces=False.
    """
    special_ids = set(tokenizer.all_special_ids)
    positions = [index for index, token in enumerate(response_ids) if token not in special_ids]
    token_ids = [response_ids[index] for index in positions]
    offsets = _byte_level_offsets(tokenizer, token_ids, output)
    if offsets is not None:
        return positions, offsets, "aligned"
    try:
        encoded = tokenizer(output, add_special_tokens=False, return_offsets_mapping=True)
    except (NotImplementedError, ValueError):
        return [], [], "offsets_unavailable"
    if encoded["input_ids"] != token_ids:
        return [], [], "token_id_mismatch"
    offsets = encoded["offset_mapping"]
    if len(offsets) != len(positions):
        raise ValueError("Tokenizer offsets do not match response token count")
    return positions, offsets, "aligned"
