#### ❓ Question:

What is the purpose of the `chunk_overlap` parameter when using `RecursiveCharacterTextSplitter` to prepare documents for RAG, and what trade-offs arise as you increase or decrease its value?

##### ✅ Answer:

The `chunk_overlap` parameter specifies how many characters from adjacent chunks should be included in each new chunk. This creates overlapping content between consecutive chunks, ensuring that sentences and context don't get cut off abruptly at chunk boundaries.

**Trade-offs:**
- **Higher overlap**: Better context preservation and smoother transitions between chunks, but increases storage overhead and potential redundancy
- **Lower overlap**: More efficient storage and less redundancy, but may lose important context at chunk boundaries and create fragmented information
- **Optimal balance**: Typically 10-20% overlap (compared to the total size of the chunk) provides good context preservation without excessive redundancy

#### ❓ Question:

Your retriever is configured with `search_kwargs={"k": 5}`. How would adjusting `k` likely affect RAGAS metrics such as Context Precision and Context Recall in practice, and why?

##### ✅ Answer:

Let's first look at the meaning of `Context Precision`:
• `Context Precision`: How well the most relevant chunks are ranked higher in retrieval results  
  └─ Measures retrieval ranking quality - are the best results at the top?
• `Context Recall`: How well the retrieval system finds relevant information from the knowledge base
  └─ Measures retrieval completeness - did we get the right documents?

Changing the value of `k` does have an impact on the both `Context Precision` and/or `Context Recall`. 

Increasing (higher) the value of `k` means we get better Context Recall, but could contain less relevant content. And also increase token usage.

While decreasing (lower) the value of `k` means we get better Context Precision, but may be missing any relevant content needed And also have reduced token usage.

#### ❓ Question:

Compare the `agent` and `agent_helpful` assistants defined in `langgraph.json`. Where does the helpfulness evaluator fit in the graph, and under what condition should execution route back to the agent vs. terminate?

##### ✅ Answer:

**`agent` (simple_agent):**
- **Flow**: START → agent → action (tool node) → END
- **Logic**: Basic tool-using agent that executes tools and terminates
- **Routing**: Agent decides whether to call tools or return final answer

**`agent_helpful` (agent_with_helpfulness):**
- **Flow**: START → agent → action (tool node) → agent → helpfulness evaluator → [conditional routing]
- **Logic**: Enhanced agent with quality control through helpfulness evaluation
- **Routing conditions**:
  - **Route back to agent**: If helpfulness evaluator determines response is NOT helpful (needs improvement), the flow continues
  - **Terminate**: If helpfulness evaluator determines response IS helpful (satisfactory quality), or until 10 iterations have been completed, in either case the flow ends

**Key difference**: The `agent_helpful` includes a feedback loop where responses are evaluated for helpfulness and can be iteratively improved before final output.
