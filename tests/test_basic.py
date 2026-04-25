# tests/test_basic.py
"""
Basic smoke tests — CI runs these on every push.
"""

def test_chunker_import():
    from smartdesk.rag.chunker import DocumentChunker
    assert DocumentChunker is not None

def test_chunker_splits_text():
    from smartdesk.rag.chunker import DocumentChunker
    chunker = DocumentChunker(chunk_size=10, overlap=2)
    chunks = chunker.chunk("one two three four five six seven eight nine ten eleven twelve", "test-doc")
    assert len(chunks) > 1

def test_embedder_import():
    from smartdesk.rag.embedder import RAGEmbedder
    assert RAGEmbedder is not None

def test_react_agent_import():
    from smartdesk.agents.react_agent import ReActAgent
    assert ReActAgent is not None

def test_tools_registry():
    from smartdesk.agents.tools import TOOL_MAP
    assert "search_invoices" in TOOL_MAP
    assert "calculate_total" in TOOL_MAP
    assert "final_answer" in TOOL_MAP