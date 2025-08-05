from pathlib import Path
import yaml
from typing import Optional, Dict, Any
from langchain_core.runnables import RunnableConfig

from agents.base_agents.base_agent import BaseAgent
from agents.utils import _load_prompt
from tools.core_tools.knowledge_search.knowledge_search_tool import knowledge_search_tool
from .waiter_tools import (
    browse_menu_api,
    check_item_availability_api,
    submit_order_to_kitchen_api,
    check_order_status_api,
    modify_order_api,
    generate_bill_api,
    process_payment_api,
    check_table_status_api,
    check_reservation_availability_api,
    log_customer_feedback_api,
    handle_special_request_api
)


class WaiterAgent(BaseAgent):
    def __init__(self,
                 prompt: Optional[str] = None,
                 *,
                 config: RunnableConfig,
                 model_kwargs: Optional[Dict[str, Any]] = None,
                 vlm_kwargs: Optional[Dict[str, Any]] = None,
                 node_kwargs: Optional[Dict[str, Any]] = None,
                 recursion_limit: Optional[int] = 25,
                 debug: bool = False):
        
        if prompt is None:
            prompt = ""
        
        waiter_agent_prompt = _load_prompt("WaiterAgent", base_dir=Path(__file__).parent)
        final_prompt = prompt + waiter_agent_prompt
        
        super().__init__(
            prompt=final_prompt,
            config=config,
            model_kwargs=model_kwargs,
            vlm_kwargs=vlm_kwargs,
            node_kwargs=node_kwargs,
            recursion_limit=recursion_limit,
            debug=debug
        )
        
        waiter_tools = [
            browse_menu_api,
            check_item_availability_api,
            submit_order_to_kitchen_api,
            check_order_status_api,
            modify_order_api,
            generate_bill_api,
            process_payment_api,
            check_table_status_api,
            check_reservation_availability_api,
            log_customer_feedback_api,
            handle_special_request_api
        ]
        
        self.bind_tools(waiter_tools)

