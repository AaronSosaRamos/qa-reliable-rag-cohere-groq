from langgraph.graph import (
    StateGraph,
    END
)

from nodes.nodes import (
    check_for_hallucinations,
    filter_non_relevant_docs,
    generate_result,
    generate_retriever,
    highlight_docs
)
from schemas.schemas import GraphState

workflow = StateGraph(GraphState)

workflow.add_node("generate_retriever", generate_retriever)
workflow.add_node("filter_non_relevant_docs", filter_non_relevant_docs)
workflow.add_node("generate_result", generate_result)
workflow.add_node("check_for_hallucinations", check_for_hallucinations)
workflow.add_node("highlight_docs", highlight_docs)

workflow.set_entry_point("generate_retriever")
workflow.add_edge("generate_retriever", "filter_non_relevant_docs")
workflow.add_edge("filter_non_relevant_docs", "generate_result")
workflow.add_edge("generate_result", "check_for_hallucinations")
workflow.add_edge("check_for_hallucinations", "highlight_docs")
workflow.add_edge("highlight_docs", END)

app = workflow.compile()