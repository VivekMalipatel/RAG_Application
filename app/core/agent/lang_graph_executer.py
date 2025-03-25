import logging
from typing import Type, Dict, Any, Callable
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END

class OmniGraph:
    """
    OmniRAG's custom LangGraph-based framework.
    Encapsulates LangGraph to define structured workflows using BaseAgent abstraction.
    """

    def __init__(self, state_schema: Type[BaseModel], graph_name: str):
        """
        Initializes OmniGraph with a structured state schema.

        Args:
            state_schema (Type[BaseModel]): Pydantic schema defining state structure.
            graph_name (str): Name of the graph.
        """
        self.state_schema = state_schema
        self.graph_name = graph_name
        self.builder = StateGraph(state_schema)
        self.nodes = {}

    def add_node(self, name: str, function: Callable):
        """
        Registers a node (function) to the graph.

        Args:
            name (str): Node name.
            function (Callable): Function to execute at the node.
        """
        self.nodes[name] = function
        self.builder.add_node(name, function)
        logging.info(f"Added node: {name}")

    def add_edge(self, from_node: str, to_node: str):
        """
        Defines a direct edge between two nodes.

        Args:
            from_node (str): Starting node.
            to_node (str): Destination node.
        """
        self.builder.add_edge(from_node, to_node)
        logging.info(f"Added edge: {from_node} → {to_node}")

    def set_entry_point(self, entry_node: str):
        """
        Defines the graph's entry point.

        Args:
            entry_node (str): Node to start execution from.
        """
        self.builder.add_edge(START, entry_node)
        logging.info(f"Set entry point: {entry_node}")

    def set_exit_point(self, exit_node: str):
        """
        Defines the graph's exit point.

        Args:
            exit_node (str): Node where execution stops.
        """
        self.builder.add_edge(exit_node, END)
        logging.info(f"Set exit point: {exit_node}")

    def compile(self):
        """
        Compiles the graph before execution.
        """
        self.graph = self.builder.compile()
        logging.info("Graph compiled successfully.")

    def execute(self, initial_state: Dict[str, Any]):
        """
        Executes the compiled graph with an initial state.

        Args:
            initial_state (Dict[str, Any]): Input data.

        Returns:
            Any: Execution result.
        """
        return self.graph.invoke(initial_state)