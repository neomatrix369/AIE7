from typing import List, Optional, TypedDict, Dict, Any, Union
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
import json

from .agents import get_default_tools
from .models import get_openai_model


class SafeAgentState(TypedDict):
    """State schema for safe agent with guardrails."""
    messages: List[BaseMessage]
    input_validated: bool
    output_validated: bool
    guard_failures: List[str]
    safety_score: float


def create_safe_langgraph_agent(
    model_name: str = "gpt-4",
    temperature: float = 0.1,
    rag_chain=None,
    enable_monitoring: bool = True
):
    """Create a production-safe LangGraph agent with completely fixed message flow.
    
    This version manually handles tool execution to avoid OpenAI API message sequencing issues.
    
    Args:
        model_name: OpenAI model name
        temperature: Model temperature
        rag_chain: Optional RAG chain to include as a tool
        enable_monitoring: Enable safety monitoring (default: True)
        
    Returns:
        Compiled safe LangGraph agent with guardrails
    """
    
    # Get tools and model
    tools = get_default_tools(rag_chain)
    model = get_openai_model(model_name=model_name, temperature=temperature)
    model_with_tools = model.bind_tools(tools)
    
    # Create tool lookup for manual execution
    tool_map = {tool.name: tool for tool in tools}
    
    def input_validation_node(state: SafeAgentState):
        """Validate user input with multiple guards."""
        if not state["messages"]:
            return {
                "input_validated": False,
                "guard_failures": ["no_input"],
                "safety_score": 0.0
            }
            
        last_message = state["messages"][-1].content
        failures = []
        
        try:
            # Import guards from main namespace (set up in notebook)
            import __main__
            
            # Check if guards are available
            if not hasattr(__main__, 'jailbreak_guard'):
                print("⚠ Warning: Guardrails not properly configured, skipping validation")
                return {
                    "input_validated": True,
                    "guard_failures": [],
                    "safety_score": 1.0
                }
                
            jailbreak_guard = __main__.jailbreak_guard
            topic_guard = __main__.topic_guard
            pii_guard = __main__.pii_guard
            
            # 1. Jailbreak detection
            jailbreak_result = jailbreak_guard.validate(last_message)
            if not jailbreak_result.validation_passed:
                failures.append("jailbreak_detected")
                
            # 2. Topic restriction  
            try:
                topic_guard.validate(last_message)
            except Exception as topic_error:
                failures.append(f"topic_validation_error: {str(topic_error)}")
            
            # 3. PII detection and redaction
            pii_result = pii_guard.validate(last_message)
            if pii_result.validated_output != last_message:
                # Update message with PII redacted version
                state["messages"][-1] = HumanMessage(content=pii_result.validated_output)
            
            return {
                "input_validated": len(failures) == 0,
                "guard_failures": failures,
                "safety_score": 1.0 if len(failures) == 0 else 0.0
            }
            
        except Exception as e:
            failures.append(f"input_validation_error: {str(e)}")
            return {
                "input_validated": False,
                "guard_failures": failures,
                "safety_score": 0.0
            }
    
    def call_agent(state: SafeAgentState):
        """Call the agent with tools."""
        messages = state["messages"]
        
        # Ensure we have valid messages
        if not messages:
            return {"messages": [AIMessage(content="I'm sorry, I didn't receive any input.")]}
            
        try:
            response = model_with_tools.invoke(messages)
            
            # Check if response has tool calls
            if hasattr(response, 'tool_calls') and response.tool_calls:
                # Execute tools manually to avoid message flow issues
                tool_messages = []
                
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_call_id = tool_call["id"]
                    
                    if tool_name in tool_map:
                        try:
                            # Execute the tool
                            tool_result = tool_map[tool_name].invoke(tool_args)
                            
                            # Create tool message
                            tool_message = ToolMessage(
                                content=str(tool_result),
                                tool_call_id=tool_call_id
                            )
                            tool_messages.append(tool_message)
                            
                        except Exception as e:
                            # Handle tool execution errors
                            error_message = ToolMessage(
                                content=f"Error executing {tool_name}: {str(e)}",
                                tool_call_id=tool_call_id
                            )
                            tool_messages.append(error_message)
                    else:
                        # Tool not found
                        error_message = ToolMessage(
                            content=f"Tool {tool_name} not found",
                            tool_call_id=tool_call_id
                        )
                        tool_messages.append(error_message)
                
                # Add the response and tool messages to the conversation
                new_messages = [response] + tool_messages
                
                # Now call the agent again to get a final response based on tool outputs
                updated_messages = messages + new_messages
                final_response = model_with_tools.invoke(updated_messages)
                
                # Return all messages including the final response
                return {"messages": new_messages + [final_response]}
            else:
                # No tool calls, just return the response
                return {"messages": [response]}
                
        except Exception as e:
            print(f"Error in agent call: {e}")
            return {"messages": [AIMessage(content="I apologize, but I encountered an error processing your request.")]}
    
    def needs_tool_response(state: SafeAgentState):
        """Check if we need to call the agent again after tool execution."""
        if not state["messages"]:
            return "output_validation"
        
        # With the simplified call_agent implementation, we always proceed to output validation
        # since tool execution and final response generation happen in a single call_agent invocation
        return "output_validation"
    
    def output_validation_node(state: SafeAgentState):
        """Validate agent output before returning."""
        if not state["messages"]:
            return {
                "output_validated": False,
                "guard_failures": ["no_response"],
                "safety_score": 0.0
            }
            
        # Find the last AI message (skip tool messages)
        last_ai_message = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage):
                last_ai_message = msg
                break
        
        if not last_ai_message or not last_ai_message.content:
            return {
                "output_validated": False,
                "guard_failures": ["no_ai_response"],
                "safety_score": 0.0
            }
            
        agent_response = last_ai_message.content
        failures = []
        
        try:
            # Import guards from main namespace
            import __main__
            
            if not hasattr(__main__, 'profanity_guard'):
                print("⚠ Warning: Output guards not properly configured, skipping validation")
                return {
                    "output_validated": True,
                    "guard_failures": [],
                    "safety_score": 1.0
                }
                
            profanity_guard = __main__.profanity_guard
            pii_guard = __main__.pii_guard
            factuality_guard = __main__.factuality_guard
            
            # 1. Content moderation
            profanity_guard.validate(agent_response)
            
            # 2. Factuality check - validate against source context
            try:
                factuality_guard.validate(agent_response)
            except Exception as factuality_error:
                failures.append(f"factuality_validation_error: {str(factuality_error)}")
            
            # 3. PII protection in output
            cleaned_response = pii_guard.validate(agent_response)
            if cleaned_response.validated_output != agent_response:
                # Update the last AI message with cleaned content
                for i in reversed(range(len(state["messages"]))):
                    if isinstance(state["messages"][i], AIMessage):
                        state["messages"][i] = AIMessage(content=cleaned_response.validated_output)
                        break
            
            return {
                "output_validated": True,
                "guard_failures": failures,
                "safety_score": 1.0
            }
            
        except Exception as e:
            failures.append(f"output_validation_error: {str(e)}")
            return {
                "output_validated": False,
                "guard_failures": failures,
                "safety_score": 0.5
            }
    
    def should_proceed_to_agent(state: SafeAgentState):
        """Route based on input validation."""
        return "call_agent" if state.get("input_validated", False) else "input_error"
    
    def should_return_response(state: SafeAgentState):
        """Route based on output validation."""
        return END if state.get("output_validated", False) else "output_error"
    
    def input_error_handler(state: SafeAgentState):
        """Handle input validation failures gracefully."""
        guard_failures = state.get("guard_failures", [])
        
        if "jailbreak_detected" in guard_failures:
            error_msg = "I cannot process requests that attempt to bypass safety guidelines. Please ask a legitimate question about student loans or financial aid."
        else:
            error_msg = "I can only help with student loan and financial aid questions. Please rephrase your question to focus on these topics."
        
        return {"messages": [AIMessage(content=error_msg)]}
    
    def output_error_handler(state: SafeAgentState):
        """Handle output validation failures with fallback."""
        fallback_msg = "I apologize, but I cannot provide a response that meets safety guidelines. Please try rephrasing your question."
        
        return {"messages": [AIMessage(content=fallback_msg)]}
    
    # Build the safe agent workflow
    workflow = StateGraph(SafeAgentState)
    
    # Add nodes
    workflow.add_node("input_validation", input_validation_node)
    workflow.add_node("call_agent", call_agent)
    workflow.add_node("output_validation", output_validation_node)
    workflow.add_node("input_error", input_error_handler)
    workflow.add_node("output_error", output_error_handler)
    
    # Add edges with proper routing
    workflow.set_entry_point("input_validation")
    workflow.add_conditional_edges(
        "input_validation", 
        should_proceed_to_agent,
        {
            "call_agent": "call_agent",
            "input_error": "input_error"
        }
    )
    workflow.add_conditional_edges(
        "call_agent", 
        needs_tool_response,
        {
            "output_validation": "output_validation"
        }
    )
    workflow.add_conditional_edges(
        "output_validation", 
        should_return_response,
        {
            END: END,
            "output_error": "output_error"
        }
    )
    workflow.add_edge("input_error", END)
    workflow.add_edge("output_error", END)
    
    return workflow.compile()


def invoke_safe_agent(agent, user_input: str) -> dict:
    """Invoke the v2 safe agent with a user input.
    
    Args:
        agent: Compiled safe agent
        user_input: User's input message
        
    Returns:
        Agent response with safety metrics
    """
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "input_validated": False,
        "output_validated": False,
        "guard_failures": [],
        "safety_score": 0.0
    }
    
    try:
        result = agent.invoke(initial_state)
        return result
    except Exception as e:
        print(f"Error in safe agent invocation: {e}")
        return {
            "messages": [AIMessage(content="I apologize, but I encountered an error processing your request.")],
            "input_validated": False,
            "output_validated": False,
            "guard_failures": [f"agent_error: {str(e)}"],
            "safety_score": 0.0
        }