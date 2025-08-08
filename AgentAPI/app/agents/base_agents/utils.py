import tiktoken
import uuid
import logging
from typing import List, Tuple, Optional
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
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
        logger.debug(f"Storing {len(trimmed_messages)} trimmed messages for background save")
        return system_messages, [], remaining_messages, True, trimmed_messages
    else:
        logger.debug("No trimming possible - keeping all state messages")
        return system_messages, [], state_messages, True, []

def get_messages_to_save(messages: List[BaseMessage]) -> List[BaseMessage]:
    second_last_ai_index = find_second_last_ai_index(messages)
    
    if second_last_ai_index != -1:
        return messages[second_last_ai_index + 1:]
    else:
        return messages

def generate_message_id(checkpoint_id: Optional[str], org_id: str, user_id: str, index: int) -> str:
    if checkpoint_id:
        return f"{checkpoint_id}_{index}"
    return f"{org_id}_{user_id}_{str(uuid.uuid4())}_{index}"

def should_trim_state(messages: List[BaseMessage]) -> bool:
    if not messages:
        return False
    
    last_message = messages[-1]
    if not hasattr(last_message, 'usage_metadata') or not last_message.usage_metadata:
        return False
    
    num_tokens = last_message.usage_metadata.get("total_tokens", 0)
    return num_tokens >= envconfig.MAX_STATE_TOKENS
