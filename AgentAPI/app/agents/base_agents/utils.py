import tiktoken
import logging
from typing import List, Tuple, Optional, Any, Set
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage
from config import config as envconfig

_ENCODING = None

def get_encoding():
    global _ENCODING
    if _ENCODING is None:
        try:
            _ENCODING = tiktoken.encoding_for_model("gpt-4")
        except KeyError:
            _ENCODING = tiktoken.get_encoding("cl100k_base")
    return _ENCODING

def count_tokens(messages: List[BaseMessage]) -> int:
    encoding = get_encoding()
    total_tokens = 0
    for message in messages:
        if hasattr(message, 'content'):
            if isinstance(message.content, str):
                total_tokens += len(encoding.encode(message.content))
            elif isinstance(message.content, list):
                for content_item in message.content:
                    if isinstance(content_item, dict) and content_item.get("type") == "text":
                        total_tokens += len(encoding.encode(content_item.get("text", "")))
    return total_tokens

def find_second_last_ai_index(messages: List[BaseMessage]) -> int:
    second_last_ai_index = -1
    ai_count = 0
    
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            ai_count += 1
            if ai_count == 2:
                second_last_ai_index = i
                break
    
    return second_last_ai_index

def trim_state_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    second_last_ai_index = find_second_last_ai_index(messages)
    
    if second_last_ai_index != -1:
        return messages[second_last_ai_index:]
    else:
        return messages

def optimize_messages_for_tokens(
    system_messages: List[BaseMessage],
    memory_messages: List[BaseMessage], 
    state_messages: List[BaseMessage],
    max_tokens: int,
    logger: Optional[logging.Logger] = None
) -> Tuple[List[BaseMessage], List[BaseMessage], List[BaseMessage], bool, List[BaseMessage]]:
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    all_messages = system_messages + memory_messages + state_messages
    total_tokens = count_tokens(all_messages)
    
    if total_tokens < max_tokens:
        return system_messages, memory_messages, state_messages, False, []
    
    logger.debug(f"Token count ({total_tokens}) exceeds limit ({max_tokens}), optimizing messages")
    
    core_messages = system_messages + state_messages
    core_tokens = count_tokens(core_messages)
    
    if core_tokens < max_tokens:
        logger.debug("Removing memory messages to fit within token limit")
        return system_messages, [], state_messages, True, []
    
    logger.debug("Trimming state messages to fit within token limit")
    second_last_ai_index = find_second_last_ai_index(state_messages)
    
    if second_last_ai_index != -1:
        trimmed_messages = state_messages[:second_last_ai_index]
        remaining_messages = state_messages[second_last_ai_index:]
        logger.debug(f"Trimming {len(trimmed_messages)} messages from state to fit token limits")
        return system_messages, [], remaining_messages, True, trimmed_messages
    else:
        logger.debug("No trimming possible - keeping all state messages")
        return system_messages, [], state_messages, True, []

def should_trim_state(messages: List[BaseMessage]) -> bool:
    if not messages:
        return False
    
    last_message = messages[-1]
    if not hasattr(last_message, 'usage_metadata') or not last_message.usage_metadata:
        return False
    
    num_tokens = last_message.usage_metadata.get("total_tokens", 0)
    return num_tokens >= envconfig.MAX_STATE_TOKENS


def get_pending_tool_calls(messages: List[Any]) -> Set[str]:
    tool_call_ids = set()
    tool_result_ids = set()
    
    for msg in messages:
        if isinstance(msg, AIMessage):
            tool_calls = getattr(msg, "tool_calls", None) or []
            for tc in tool_calls:
                if isinstance(tc, dict):
                    tc_id = tc.get("id")
                else:
                    tc_id = getattr(tc, "id", None)
                if tc_id:
                    tool_call_ids.add(tc_id)
        
        elif isinstance(msg, ToolMessage):
            tc_id = getattr(msg, "tool_call_id", None)
            if tc_id:
                tool_result_ids.add(tc_id)
    
    pending = tool_call_ids - tool_result_ids
    return pending
