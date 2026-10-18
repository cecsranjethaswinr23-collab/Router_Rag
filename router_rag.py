import os
import argparse
from dotenv import load_dotenv
from typing import List, Literal, TypedDict
from pydantic import BaseModel, Field
from duckduckgo_search import DDGS
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

load_dotenv()
# ==============================================================================
# 1. State Definition & Router Schema
# ==============================================================================
class AgentState(TypedDict):
    query: str
    route: str
    context: str
    final_answer: str

class RoutingDecision(BaseModel):
    destination: Literal["db_insurance", "db_motor_laws", "web_search", "direct_chat"] = Field(...,
                                                                                               description=("Route to 'db_insurance' for insurance policies, claims, or coverage of the companies like Star, United India, HDFC, Care. "
                                                                                                            "Route to 'db_motor_laws' for legal sections, traffic violations, and vehicles do's and dont's. "
                                                                                                            "Route to 'web_search' for live, dynamic data like stock prices, current gold rates, or recent news that are non-conversational queries "
                                                                                                            "Route to 'direct_chat' for basic greetings (e.g., 'how are you', how was the weather) or general conversational questions and simple general questions in day to day life humans would know."
                                                                                                            )
                                                                                                )

# ==============================================================================
# 2. Initialize Embeddings & Vector Databases
# ==============================================================================
print("-----Loading Embeddings & Databases-----")
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")



# Mocking the local FAISS databases for demonstration. 
db_insurance = FAISS.load_local("faiss_index",hf_embeddings,allow_dangerous_deserialization=True)
db_motor_laws =  FAISS.load_local("laws_faiss_index",hf_embeddings,allow_dangerous_deserialization=True)

# Initialize the LLM (Gemini used here as the engine)
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.1)
router_llm = llm.with_structured_output(RoutingDecision)

# ==============================================================================
# 3. Define Graph Nodes
# ==============================================================================
def router_node(state: AgentState) -> dict:
    """The LLM analyzes the query and decides the route."""
    decision = router_llm.invoke([
        SystemMessage(content="You are a smart routing agent. Classify the user query based on the contents actually mentioned like a logical human."),
        HumanMessage(content=state["query"])
    ])
    return {"route": decision.destination}

def retrieve_insurance_node(state: AgentState) -> dict:
    docs = db_insurance.similarity_search(state["query"], k=4)
    context = "\n---\n".join([d.page_content for d in docs])
    return {"context": f"[INSURANCE DB]\n{context}"}

def retrieve_motor_laws_node(state: AgentState) -> dict:
    docs = db_motor_laws.similarity_search(state["query"], k=4)
    context = "\n---\n".join([d.page_content for d in docs])
    return {"context": f"[MOTOR LAWS DB]\n{context}"}

def web_search_node(state: AgentState) -> dict:
    results = DDGS().text(state["query"], max_results=4)
    context = "\n---\n".join([f"{r['title']}: {r['body']}" for r in results])
    return {"context": f"[WEB SEARCH]\n{context}"}

def aggregator_node(state: AgentState) -> dict:
    """Invokes the LLM to generate the final answer using retrieved context (if any)."""
    if state["route"] == "direct_chat":
        prompt = f"Respond conversationally to this user input: '{state['query']}'"
    else:
        prompt = f"""Use the provided context to answer the user's query accurately. 
        If the context does not contain the answer, say you don't know.
        
        Context:
        {state['context']}
        
        User Query: {state['query']}
        Answer:"""
    
    response = llm.invoke(prompt)
    return {"final_answer": response.content}

# ==============================================================================
# 4. Build the Graph Workflow
# ==============================================================================
def route_condition(state: AgentState) -> str:
    return state["route"]

workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("Router", router_node)
workflow.add_node("Retrieve_Insurance", retrieve_insurance_node)
workflow.add_node("Retrieve_Motor_Laws", retrieve_motor_laws_node)
workflow.add_node("Web_Search", web_search_node)
workflow.add_node("Aggregator", aggregator_node)

# Add Edges
workflow.add_edge(START, "Router")

# Conditional Routing from the Router Node
workflow.add_conditional_edges(
    "Router",
    route_condition,
    {
        "db_insurance": "Retrieve_Insurance",
        "db_motor_laws": "Retrieve_Motor_Laws",
        "web_search": "Web_Search",
        "direct_chat": "Aggregator" # Skips retrieval completely
    }
)

# Funnel all retrievers into the Aggregator
workflow.add_edge("Retrieve_Insurance", "Aggregator")
workflow.add_edge("Retrieve_Motor_Laws", "Aggregator")
workflow.add_edge("Web_Search", "Aggregator")

# End the process after aggregation
workflow.add_edge("Aggregator", END)

# Compile the graph
app = workflow.compile()

# ==============================================================================
# 5. Execution Test
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the Agentic Router RAG.")
    parser.add_argument("--query", "-q", type=str, required=True, help="Question to ask")
    args = parser.parse_args()

    result = app.invoke({"query": args.query})

    raw_answer = result.get("final_answer", "")

    # Safely convert list responses into a single string
    if isinstance(raw_answer, list):
        answer_text = "\n".join(
            item.get("text", str(item)) if isinstance(item, dict) else str(item)
            for item in raw_answer
        )
    else:
        answer_text = str(raw_answer)

    print(f"\n[Route]: {result.get('route')}")
    print(f"[Answer]:\n{answer_text.strip()}\n")