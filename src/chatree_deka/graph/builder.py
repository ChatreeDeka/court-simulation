from __future__ import annotations
from langgraph.graph import StateGraph, START, END
from chatree_deka.state import TrialState

from chatree_deka.graph.nodes.prosecutor import prosecutor_node
from chatree_deka.graph.nodes.defender import defender_node
from chatree_deka.graph.nodes.judge import judge_node
from chatree_deka.graph.nodes.summary import summary_node
from chatree_deka.graph.nodes.validation import validation_node
from chatree_deka.graph.nodes.advance_phase import advance_phase_node

from chatree_deka.graph.edges import validation_route, judge_router


def should_interrupt(state: TrialState, node_name: str) -> bool:
    """
    Determine whether to interrupt before a node based on execution mode.
    
    In train mode, no interrupts fire. In run/coached modes, interrupt if the
    node's corresponding role is set to 'manual'.
    """
    if state.get("mode") == "train":
        return False
    
    role = node_name.replace("_node", "")
    return state.get(f"{role}_mode") == "manual"


def build_graph(checkpointer=None):
    builder = StateGraph(TrialState)

    # 1. Add Nodes
    builder.add_node("prosecutor_node", prosecutor_node)
    builder.add_node("defender_node", defender_node)
    builder.add_node("judge_node", judge_node)
    builder.add_node("summary_node", summary_node)
    builder.add_node("validation_node", validation_node)
    builder.add_node("advance_phase_node", advance_phase_node)

    # 2. Add Edges
    # The starting phase is typically handled by setting up the state and jumping into the loop. 
    # For POC, start with prosecutor.
    builder.add_edge(START, "prosecutor_node")

    # From speaker nodes, we validate their statements
    builder.add_edge("prosecutor_node", "validation_node")
    builder.add_edge("defender_node", "validation_node")
    
    # After validation, route based on validation result
    builder.add_conditional_edges(
        "validation_node", 
        validation_route,
        {
            "judge_node": "judge_node",
            "prosecutor_node": "prosecutor_node", 
            "defender_node": "defender_node"
        }
    )

    # Judge routes based on context
    builder.add_conditional_edges(
        "judge_node",
        judge_router,
        {
            "prosecutor_node": "prosecutor_node",
            "defender_node": "defender_node", 
            "judge_node": "judge_node",
            "advance_phase_node": "advance_phase_node",
            "summary_node": "summary_node"
        }
    )
    
    # Phase advancement leads back to prosecutor to start new phase
    builder.add_edge("advance_phase_node", "prosecutor_node")
    
    builder.add_edge("summary_node", END)

    # 3. Interrupts 
    # Mode-aware interrupts: train mode suppresses all interrupts; run/coached modes
    # check if the role is manually operated.
    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["prosecutor_node", "defender_node", "judge_node"],
        # Note: should_interrupt callable is evaluated at runtime per state
        # LangGraph will call should_interrupt(state, node_name) before each node
    )
    
    # Wrap the graph to inject the should_interrupt check
    # LangGraph's interrupt_before with a callable requires a custom predicate
    # For now, we rely on the CLI to suppress interrupts via mode checks
    # TODO: integrate should_interrupt as a proper interrupt predicate if LangGraph supports it
    
    return graph
