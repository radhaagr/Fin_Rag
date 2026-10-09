"""
LangGraph Router Agent — Customer Support Example
---------------------------------------------------
Flow: User Message -> Receive -> Classify -> Router -> (Billing / Technical / General) -> END
 
Install:
    pip install langgraph langchain-openai
 
Set your API key (get one at https://platform.openai.com/api-keys):
    export OPENAI_API_KEY=sk-...      (macOS/Linux)
    setx OPENAI_API_KEY "sk-..."      (Windows, then restart terminal)
 
Run:
    python router_agent.py
"""
import time
from typing import TypedDict, Literal
 
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv
load_dotenv()  

router_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0, max_tokens=512)
support_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3, max_tokens=1024)


#-----------------------------------------------------------------------
class SupportState(TypedDict):
    user_message: str
    recieved_at: float
    message_type:str
    response: str

def recieve_message(state: SupportState) -> SupportState:
    return { **state,
              "user_message": state['user_message'].strip(),
              "recieved_at": time.time(), }

class Classification(   BaseModel):
    message_type: Literal["billing", "technical", "general"]


def classify_message(state: SupportState) -> SupportState:
    classifier = router_llm.with_structured_output(Classification)
    result =  classifier.invoke( f"""
    Classify this customer messages into exactly one of three categories: 
    - billing: questions about billing, invoices, or payments.
    - technical: questions about product functionality or troubleshooting.
    - general: questions that don't fit into the other categories.
    
    Message: "{state['user_message']}"""
    )
    return {**state, "message_type": result.message_type}


def router(state: SupportState) -> SupportState:
    """Pure pass-through — routing decision itself lives in route_decision()."""
    return state

def route_decision(state: SupportState) -> str:
    return state["message_type"]  # "billing" | "technical" | "general"

def billing_support(state: SupportState) -> SupportState:
    answer = support_llm.invoke(f"""
    You are a billing support agent. Respond to the following customer message in a helpful and professional manner.
    
    Customer Message: "{state['user_message']}"
    """)
    return {**state, "response": answer.content}

def technical_support(state: SupportState) -> SupportState:
    answer = support_llm.invoke(f"""
    You are a technical support agent. Respond to the following customer message in a helpful and professional manner.
    
    Customer Message: "{state['user_message']}"
    """)
    return {**state, "response": answer.content}

def general_support(state: SupportState) -> SupportState:
    answer = support_llm.invoke(f"""
    You are a general support agent. Respond to the following customer message in a helpful and professional manner.
    
    Customer Message: "{state['user_message']}"
    """)
    return {**state, "response": answer.content}

graph = StateGraph(SupportState)
graph.add_node("receive_message", recieve_message)
graph.add_node("classify_message", classify_message)
graph.add_node("router", router)
graph.add_node("billing_support", billing_support)
graph.add_node("technical_support", technical_support)
graph.add_node("general_support", general_support)

graph.set_entry_point("receive_message")

graph.add_edge("receive_message", "classify_message")
graph.add_edge("classify_message", "router")

graph.add_conditional_edges("router", route_decision, {
                                "billing": "billing_support",
                                "technical": "technical_support",
                                "general": "general_support",
                            })

graph.add_edge("billing_support", END)
graph.add_edge("technical_support", END)
graph.add_edge("general_support", END)

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

def main():
    print("Support agent ready. Type 'exit' or 'quit' to stop.\n")
    print("---------------------------------------------------")
    config = {"configurable": {"thread_id": "cli-session-1"}}
    while True:
        user_input  = input("\nCustomer Message: ")
        if user_input.strip().lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if not user_input.strip():
            continue
        
        result = app.invoke({"user_message": user_input}, config=config)
        print(f"\nSupport Response: {result['response']}")

if __name__ == "__main__":
    main()





