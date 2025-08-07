
⏺ Open Deep Research Notebook Summary

  Overview

  • Purpose: Automated AI research agent that generates comprehensive technical reports
  • Architecture: Uses LangGraph to orchestrate multi-step research and writing workflow
  • Input: Takes any topic and produces structured reports with citations

  Core Components

  State Management

  • Two-level structure: Report-level (topic, sections, final report) and Section-level (queries, sources, content)
  • Tracks progress: Search iterations, completed sections, research content
  • Uses Pydantic models: For structured data validation and type safety

  Search Capabilities

  • Multiple APIs supported: Tavily, Perplexity, Exa, ArXiv, PubMed
  • Concurrent processing: Async functions for parallel web searches
  • Source handling: Deduplication, formatting, and content limits per source
  • Academic focus: Specialized functions for ArXiv papers and PubMed research

  Configuration System

  • Flexible models: Supports Anthropic, OpenAI, Groq providers
  • Search parameters: Configurable query limits, search depth, API settings
  • Provider switching: Easy toggle between different AI models for planning vs writing

  Workflow Process

  Planning Phase

  • Initial plan generation: Creates report structure with sections
  • Human feedback loop: User approves or provides revision feedback
  • Query generation: Creates targeted search queries for each section

  Research Phase

  • Parallel section building: Uses Send() API for concurrent processing
  • Iterative search: Multiple rounds of queries and source gathering
  • Quality control: Reflection/grading system to assess section completeness

  Writing Phase

  • Section writing: Generates 150-200 word sections with citations
  • Final sections: Writes intro/conclusion using completed research
  • Report compilation: Combines all sections into final document

  Execution Features

  • Checkpointing: Resumable execution with memory persistence
  • Interrupt handling: Human approval points in the workflow
  • Error recovery: Graceful handling of API failures and rate limits
  • Structured output: Markdown formatting with proper citations

  Example Output

  • Topic: "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"
  • Generated: 5-section technical report with research citations
  • Quality: Professional technical writing with structured analysis