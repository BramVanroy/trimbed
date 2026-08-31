"""The GPT-2-style byte-to-unicode mapping used by ByteLevel tokenizers."""

from __future__ import annotations

from functools import cache


@cache
def bytes_to_unicode() -> dict[int, str]:
    """Return the reversible byte -> printable-character map used by ByteLevel.

    Borrowed mostly from GPT-2's tokenizer, also see the [transformers implementation](https://github.com/huggingface/transformers/blob/36bc98ef9dd009569366f5e253ec1876ecafd925/src/transformers/convert_slow_tokenizer.py#L1879).

    Returns:
        A mapping of all 256 byte values to distinct printable characters, e.g.
        `32 -> "Ġ"` (space), `10 -> "Ċ"` (newline), `0 -> "Ā"`, and `65 -> "A"`
        for the bytes that are already printable.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    mapped = list(printable)
    spare = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            mapped.append(256 + spare)
            spare += 1
    return {byte: chr(code) for byte, code in zip(printable, mapped, strict=True)}


@cache
def unicode_to_bytes() -> dict[str, int]:
    """Return the inverse of [`bytes_to_unicode`][trimbed.bytelevel.bytes_to_unicode].

    E.g. `"Ġ" -> 32` and `"Ċ" -> 10`.
    """
    return {char: byte for byte, char in bytes_to_unicode().items()}


@cache
def byte_level_alphabet() -> frozenset[str]:
    """Return the 256 characters a ByteLevel pre-tokenizer can produce.

    Every one of them must stay in the vocabulary of a byte-level tokenizer, otherwise
    some byte sequences become unencodable.
    """
    return frozenset(bytes_to_unicode().values())


def decode_byte_level(token: str) -> str | None:
    """Turn a byte-level token back into the text it represents.

    Args:
        token: A token as stored in a ByteLevel vocabulary, e.g. `"Ġde"`, which will
            produce `" de"`. Multi-byte characters arrive as several alphabet
            characters, so `"Ã©"` produces `"é"`.

    Returns:
        The decoded text, or `None` if the token is a partial UTF-8 sequence or
        contains characters outside the byte-level alphabet.
    """
    table = unicode_to_bytes()
    try:
        raw = bytes(table[char] for char in token)
    except KeyError:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
