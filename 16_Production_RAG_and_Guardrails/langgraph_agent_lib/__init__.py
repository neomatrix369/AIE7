"""LangGraph Agent Library

A library for LangGraph agents with caching, monitoring, guardrails, and agent integration.
"""

from .agents import create_langgraph_agent
from .caching import CacheBackedEmbeddings, setup_llm_cache
from .rag import ProductionRAGChain
from .models import get_openai_model
from .agents_with_guardrails import create_safe_langgraph_agent, invoke_safe_agent

__version__ = "0.1.0"
__all__ = [
    "create_langgraph_agent",
    "CacheBackedEmbeddings",
    "setup_llm_cache",
    "ProductionRAGChain",
    "get_openai_model",
    "create_safe_langgraph_agent",
    "invoke_safe_agent"
]

