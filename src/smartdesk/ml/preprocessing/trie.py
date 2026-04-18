"""Trie (prefix tree) for O(L) word lookup.

Why Trie in an NLP pipeline?
  - Stop-word dictionaries with 10K+ entries: Trie beats set for prefix queries.
  - PII scrubbing: match thousands of bad patterns in one pass (Aho-Corasick extends this).
  - Autocomplete, spell-check, domain-specific dictionaries.

Complexity:
  - Insert: O(L) where L = word length
  - Search: O(L)
  - Space: O(total chars across all words) — but shared prefixes save memory.

Interview questions this covers:
  - "Implement a Trie" (LeetCode 208 — very common)
  - "Word Search II" (LeetCode 212)
  - "Replace Words" (LeetCode 648 — literally this use case)
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrieNode:
    """One node in the Trie. Children is a hashmap: char -> TrieNode."""

    children: dict[str, TrieNode] = field(default_factory=dict)
    is_word_end: bool = False
    # In production we often store extra data at word ends — e.g. POS tag, category
    payload: object | None = None


class Trie:
    """Prefix tree for fast word lookup."""

    def __init__(self) -> None:
        self.root = TrieNode()
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def insert(self, word: str, payload: object | None = None) -> None:
        """Insert a word. Optional payload stored at the end node."""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        if not node.is_word_end:
            self._size += 1
        node.is_word_end = True
        node.payload = payload

    def contains(self, word: str) -> bool:
        """Return True if word exists in Trie."""
        node = self._traverse(word)
        return node is not None and node.is_word_end

    def starts_with(self, prefix: str) -> bool:
        """Return True if any word starts with prefix."""
        return self._traverse(prefix) is not None

    def _traverse(self, s: str) -> TrieNode | None:
        """Walk down the trie following s. Return the final node, or None."""
        node = self.root
        for char in s:
            if char not in node.children:
                return None
            node = node.children[char]
        return node

    @classmethod
    def from_words(cls, words: list[str]) -> Trie:
        """Bulk-load a Trie from a word list."""
        trie = cls()
        for w in words:
            trie.insert(w)
        return trie
