import logging
from typing import Dict, Any
from app.core.agent.base_agent import BaseAgent
from app.core.agent.lang_graph_executer import OmniGraph

class AgentExecutor:
    """
    Manages the execution of BaseAgents within OmniGraph.
    """

    def __init__(self, graph: OmniGraph):
        self.graph = graph
        self.agents = {}

    def register_agent(self, agent: BaseAgent, node_name: str):
        """
        Registers an agent as a node in OmniGraph.

        Args:
            agent (BaseAgent): The agent instance to be registered.
            node_name (str): The name of the node in the graph.
        """
        async def agent_wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
            return await agent.run(state)

        self.graph.add_node(node_name, agent_wrapper)
        self.agents[node_name] = agent

    def execute_graph(self, initial_state: Dict[str, Any]):
        return self.graph.execute(initial_state)