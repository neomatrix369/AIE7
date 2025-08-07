# Improved RAG Prompt for Better Context Relevance Handling

from langchain_core.prompts import ChatPromptTemplate

IMPROVED_HUMAN_TEMPLATE = """
#CONTEXT:
{context}

#QUERY:
{query}

#INSTRUCTIONS:
You are a helpful assistant that answers questions based ONLY on the provided context. Follow these steps:

1. **RELEVANCE CHECK**: First, evaluate whether the provided context is relevant to answering the query. Consider:
   - Does the context contain information that directly addresses the query?
   - Are the topics, entities, or concepts in the context related to the query?
   - Is the context from the same domain or subject area as the query?

2. **RESPONSE DECISION**:
   - If the context is RELEVANT and contains sufficient information: Provide a comprehensive answer using only the provided context
   - If the context is IRRELEVANT or unrelated to the query: Respond with "I don't have relevant information to answer this question. The provided context is not related to your query."
   - If the context is PARTIALLY RELEVANT but insufficient: Respond with "I have some related information, but it's not sufficient to fully answer your question. [Briefly mention what relevant information exists]"

3. **IMPORTANT RULES**:
   - NEVER make up information not present in the context
   - NEVER use external knowledge to supplement the context
   - If uncertain about relevance, err on the side of caution and indicate the context is not relevant
   - Be specific about why the context is or isn't relevant

#EXAMPLES:
- Query: "What is the capital of France?" with context about loan programs → "I don't have relevant information to answer this question. The provided context is not related to your query."
- Query: "What are the loan limits?" with context about loan programs → Provide detailed answer from context
- Query: "What are the loan limits?" with context about cooking recipes → "I don't have relevant information to answer this question. The provided context is not related to your query."

Now, analyze the provided context and query according to these instructions.
"""

# Alternative shorter version for token efficiency
CONCISE_HUMAN_TEMPLATE = """
#CONTEXT:
{context}

#QUERY:
{query}

#INSTRUCTIONS:
Answer the query using ONLY the provided context. 

**RELEVANCE CHECK**: First determine if the context is relevant to the query:
- RELEVANT: Context directly addresses the query → Provide detailed answer
- IRRELEVANT: Context is unrelated to query → Respond: "I don't have relevant information to answer this question. The provided context is not related to your query."
- PARTIALLY RELEVANT: Some related info but insufficient → Respond: "I have some related information but it's not sufficient to fully answer your question."

**RULES**: Never use external knowledge. If uncertain, indicate context is not relevant.
"""

# Create the prompt templates
improved_chat_prompt = ChatPromptTemplate.from_messages([
    ("human", IMPROVED_HUMAN_TEMPLATE)
])

concise_chat_prompt = ChatPromptTemplate.from_messages([
    ("human", CONCISE_HUMAN_TEMPLATE)
])

# Usage example:
# generator_chain = improved_chat_prompt | openai_chat_model | StrOutputParser() 