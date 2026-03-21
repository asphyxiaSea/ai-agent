from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.workflows.vegetation_analysis.nodes import (
    index_metrics_node,
    mask_build_node,
    sam_segment_node,
)
from app.workflows.vegetation_analysis.state import VegetationAnalysisState


def build_vegetation_analysis_graph():
    graph_builder = StateGraph(VegetationAnalysisState)
    graph_builder.add_node("sam_segment", sam_segment_node)
    graph_builder.add_node("mask_build", mask_build_node)
    graph_builder.add_node("index_metrics", index_metrics_node)

    graph_builder.add_edge(START, "sam_segment")
    graph_builder.add_edge("sam_segment", "mask_build")
    graph_builder.add_edge("mask_build", "index_metrics")
    graph_builder.add_edge("index_metrics", END)

    return graph_builder.compile()
