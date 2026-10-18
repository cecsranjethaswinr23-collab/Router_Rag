import os
from typing import List, Literal, TypedDict
from pydantic import BaseModel, Field
from duckduckgo_search import DDGS
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

# ==============================================================================
# 1. State Definition & Router Schema
# ==============================================================================
class AgentState(TypedDict):
    query: str
    route: str
    context: str
    final_answer: str

class RoutingDecision(BaseModel):
    destination: Literal["db_insurance", "db_motor_laws", "web_search", "direct_chat"] = Field(
        ...,
        description=(
            "Route to 'db_insurance' for insurance policies, claims, or coverage. "
            "Route to 'db_motor_laws' for legal sections, traffic violations, and acts. "
            "Route to 'web_search' for live, dynamic data like stock prices, current gold rates, or recent news. "
            "Route to 'direct_chat' for basic greetings (e.g., 'how are you') or general conversational questions."
        )
    )

# ==============================================================================
# 2. Initialize Embeddings & Vector Databases
# ==============================================================================
print("Loading Embeddings & Databases...")
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Mocking the local FAISS databases for demonstration. 
# In production, use: FAISS.load_local("path/to/db", hf_embeddings)
db_insurance = FAISS.from_texts(["Zero depreciation covers 100% of plastic parts."], hf_embeddings)
db_motor_laws = FAISS.from_texts(["Section 194D mandates a Rs. 2000 fine for driving without insurance."], hf_embeddings)

# Initialize the LLM (Gemini used here as the engine)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
router_llm = llm.with_structured_output(RoutingDecision)

# ==============================================================================
# 3. Define Graph Nodes
# ==============================================================================
def router_node(state: AgentState) -> dict:
    """The LLM analyzes the query and decides the route."""
    decision = router_llm.invoke([
        SystemMessage(content="You are a smart routing agent. Classify the user query."),
        HumanMessage(content=state["query"])
    ])
    return {"route": decision.destination}

def retrieve_insurance_node(state: AgentState) -> dict:
    docs = db_insurance.similarity_search(state["query"], k=2)
    context = "\n---\n".join([d.page_content for d in docs])
    return {"context": f"[INSURANCE DB]\n{context}"}

def retrieve_motor_laws_node(state: AgentState) -> dict:
    docs = db_motor_laws.similarity_search(state["query"], k=2)
    context = "\n---\n".join([d.page_content for d in docs])
    return {"context": f"[MOTOR LAWS DB]\n{context}"}

def web_search_node(state: AgentState) -> dict:
    results = DDGS().text(state["query"], max_results=3)
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

    print(f"\n[Route]: {result.get('route')}")
    print(f"[Answer]:\n{result.get('final_answer', '').strip()}\n")