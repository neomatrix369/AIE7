
  High-Level Flow & Purpose (30 seconds)

  - What it is: An unrolled implementation of Open Deep Research - an AI agent that automatically generates comprehensive
  technical reports
  - Purpose: Takes a topic, creates a research plan, conducts web searches, and writes structured reports with citations
  - Architecture: Uses LangGraph for orchestrating multi-step research and writing workflow
  - Key Update: Now shows the complete execution with an actual generated report

  Key Areas to Cover:

  1. Visual Architecture (30 seconds) - Cell 1

  - Point to the workflow diagram image that shows the complete flow
  - Highlight the visual representation of the multi-step process from topic to final report

  2. State Management (30 seconds) - Cells 7-8

  - Show the hierarchical state structure: Report-level and Section-level states
  - Explain how Annotated[list, operator.add] accumulates completed sections from parallel processes
    *     ⏺ The line Annotated[list, operator.add] is a LangGraph-specific annotation that tells the framework how to handle state
    updates when multiple nodes try to modify the same field.

    What it means:

    • Annotated: Python typing annotation that adds metadata to a type
    • list: The actual data type (a list)
    • operator.add: The function that defines how to combine/merge values

    What it does:

    • Normal behavior: If two nodes update the same state field, the second one overwrites the first
    • With operator.add: Instead of overwriting, it concatenates/combines the values
    • For lists: operator.add means list concatenation ([1,2] + [3,4] = [1,2,3,4])

    In this notebook context:

    completed_sections: Annotated[list, operator.add]

    • Multiple section nodes run in parallel and each completes a section
    • Without annotation: Only the last completed section would be saved
    • With operator.add: All completed sections get accumulated/combined into one list
    • Result: The final state contains ALL completed sections from ALL parallel nodes

    Simple example:

    # Node A completes: completed_sections = [section1]
    # Node B completes: completed_sections = [section2]
    # Node C completes: completed_sections = [section3]

    # Without operator.add: final result = [section3] (overwrites)
    # With operator.add: final result = [section1, section2, section3] (accumulates)

    This is essential for the parallel section building to work correctly - otherwise you'd lose most of your research results!

  - Point out the use of Pydantic models for structured data validation

  3. Multi-API Search Integration (45 seconds) - Cells 19-28

  - Highlight the 5 different search APIs: Tavily, Perplexity, Exa, ArXiv, PubMed
  - Show tavily_search_async function as example of concurrent search execution
  - Demonstrate source formatting and deduplication utilities for clean results

  4. Graph Architecture & Execution (60 seconds) - Cells 33-52, 54-59

  - Core workflow: Plan → Human Feedback → Research → Write → Compile
  - Show key nodes with their specific purposes:
    - generate_report_plan: Creates initial structure with web research
    - human_feedback: Human-in-the-loop approval/revision system
    - build_section_with_web_research: Parallel section research with quality control
    - write_section: Content generation with reflection and iterative improvement
  - Point out the nested graph structure for section building
  - Show the checkpointing system for resumable execution

  5. Live Execution Results (45 seconds) - Cell 59 output

  - Show the actual generated report on "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"
  - Highlight the professional structure: Introduction → Technical Sections → Conclusion
  - Point to the quality of citations and technical writing produced
  - Show how the interrupt mechanism worked for human approval

  Key Technical Highlights to Mention:
  - Async/concurrent processing for efficiency across multiple search APIs
  - Quality control loop with reflection/grading for content improvement
  - Configurable models and APIs (easily switch between Anthropic, OpenAI, etc.)
  - Human-in-the-loop approval process with resumable execution
  - Real-world output quality - professional technical reports with proper citations

  Major Update from Previous Version:
  - Now shows complete end-to-end execution with actual report generation
  - Demonstrates the interrupt and resume functionality in action
  - Shows real output quality rather than just theoretical architecture

  This covers the essential flow from topic input → planning → human approval → research → writing → final report,
  demonstrating both the technical sophistication and the practical, high-quality results the system produces.
