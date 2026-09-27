"""Registry of every RAG type. Folders are numbered for reading order, so they are loaded by path."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parent

# short name -> (folder, class name)
RAG_TYPES = {
    "naive": ("01_naive_rag", "NaiveRAG"),
    "advanced": ("02_advanced_rag", "AdvancedRAG"),
    "hybrid": ("03_hybrid_rag", "HybridRAG"),
    "multi_query": ("04_multi_query_rag", "MultiQueryRAG"),
    "hyde": ("05_hyde_rag", "HyDERAG"),
    "parent_document": ("06_parent_document_rag", "ParentDocumentRAG"),
    "contextual_retrieval": ("07_contextual_retrieval_rag", "ContextualRetrievalRAG"),
    "conversational": ("08_conversational_rag", "ConversationalRAG"),
    "corrective": ("09_corrective_rag", "CorrectiveRAG"),
    "self_rag": ("10_self_rag", "SelfRAG"),
    "adaptive": ("11_adaptive_rag", "AdaptiveRAG"),
    "agentic": ("12_agentic_rag", "AgenticRAG"),
    "graph": ("13_graph_rag", "GraphRAG"),
    "citation": ("14_citation_rag", "CitationRAG"),
}


def load_rag_class(name: str):
    folder, cls_name = RAG_TYPES[name]
    spec = importlib.util.spec_from_file_location(f"rag_types.{folder}.pipeline", _ROOT / folder / "pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, cls_name)
