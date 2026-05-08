from __future__ import annotations
import operator
from langgraph.graph import StateGraph, START, END
from chatree_deka.state import TrialState

from chatree_deka.graph.nodes.prosecutor import prosecutor_node
from chatree_deka.graph.nodes.defender import defender_node
from chatree_deka.graph.nodes.judge import judge_node
from chatree_deka.graph.nodes.evaluator import evaluator_node
from chatree_deka.graph.nodes.validation import validation_node

from chatree_deka.graph.edges import validation_route, phase_router_after_val, post_judge_router

def build_graph(checkpointer=None):
    builder = StateGraph(TrialState)

    # 1. Add Nodes
    builder.add_node("prosecutor_node", prosecutor_node)
    builder.add_node("defender_node", defender_node)
    builder.add_node("judge_node", judge_node)
    builder.add_node("evaluator_node", evaluator_node)
    builder.add_node("validation_node", validation_node)

    # 2. Add Edges
    # The starting phase is typically handled by setting up the state and jumping into the loop. 
    # For POC, start with prosecutor.
    builder.add_edge(START, "prosecutor_node")

    # From speaker nodes, we validate their statements
    builder.add_edge("prosecutor_node", "validation_node")
    builder.add_edge("defender_node", "validation_node")
    
    # After validation, we route
    
    # Then the phase router connects after successful validation
    # Actually validation_route goes to 'phase_router_after_val' for a pass
    # which is a conditional mapping. We can use add_conditional_edges out of a dummy or redirect.
    # To keep it standard, validation conditional returns the string naming the next node!
    builder.add_conditional_edges(
        "validation_node", 
        validation_route,
        {
            "phase_router_after_val": "phase_router_after_val", 
            "prosecutor_node": "prosecutor_node", 
            "defender_node": "defender_node",
            "judge_node": "judge_node"
        }
    )

    # Note: StateGraph conditional edges require mapping. We need a dummy node for phase routing
    # OR we can just embed phase routing inside validation_route when valid. 
    # To follow the separation in edges.py:
    def phase_routing_edge(state):
        return phase_router_after_val(state)
        
    builder.add_node("phase_router_after_val", lambda state: {}) # No-op node
    builder.add_conditional_edges("phase_router_after_val", phase_routing_edge)

    builder.add_conditional_edges("judge_node", post_judge_router)
    builder.add_edge("evaluator_node", END)

    # 3. Interrupts 
    # State validation for AI/Manual is evaluated in the CLI runtime loop.
    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["prosecutor_node", "defender_node", "judge_node"]
    )
    return graph
