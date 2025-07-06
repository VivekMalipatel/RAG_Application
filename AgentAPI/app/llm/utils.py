from typing import List, Dict, Any
import asyncio
import logging
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_core.messages.content_blocks import is_data_content_block
from langchain_core.language_models.chat_models import BaseChatModel
from config import config


def has_images(messages: List[BaseMessage]) -> bool:
    return has_media(messages, media_types=["image"])


def has_media(messages: List[BaseMessage], media_types: List[str] = None) -> bool:
    if media_types is None:
        media_types = ["image", "audio", "video"]
    
    if not messages:
        return False
    
    logger = logging.getLogger(__name__)
    error_count = 0
    
    for message in messages:
        try:
            if message_contains_media(message, media_types):
                return True
        except Exception as e:
            error_count += 1
            if error_count <= 3:
                logger.debug(f"Error checking message for media: {e}")
            continue
    
    if error_count > 3:
        logger.warning(f"Encountered {error_count} errors while checking messages for media")
    
    return False


def message_contains_media(message: BaseMessage, media_types: List[str]) -> bool:
    if not isinstance(message, BaseMessage):
        raise ValueError(f"Expected BaseMessage, got {type(message)}")
    
    content = getattr(message, "content", None)
    
    if isinstance(content, str):
        return any(f"data:{media_type}" in content for media_type in media_types)
    
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                if is_data_content_block(block):
                    if block["type"] in media_types:
                        return True
                    if block["type"] == "file":
                        mime_type = block.get("mime_type", "")
                        if mime_type.startswith(tuple(f"{m}/" for m in media_types)):
                            return True
                elif is_media_block(block, media_types):
                    return True
    
    if isinstance(message, ToolMessage):
        artifact = getattr(message, "artifact", None)
        if isinstance(artifact, dict):
            if artifact.get("type") in media_types:
                return True
            if any(k in artifact for k in ("base64_data", "data")):
                return True
    
    additional_kwargs = getattr(message, "additional_kwargs", None)
    if isinstance(additional_kwargs, dict):
        for key, value in additional_kwargs.items():
            if isinstance(value, str):
                if any(f"data:{media_type}" in value for media_type in media_types):
                    return True
            elif isinstance(value, dict):
                if is_media_block(value, media_types):
                    return True
    
    return False


def is_media_block(block: dict, media_types: List[str]) -> bool:
    if not isinstance(block, dict):
        return False
    
    block_type = block.get('type', '')
    
    if block_type in media_types:
        return True
    
    if block_type in ['image_url', 'audio_url', 'video_url']:
        media_type = block_type.split('_')[0]
        return media_type in media_types
    
    if block_type == 'file':
        mime_type = block.get('mime_type', '')
        return any(mime_type.startswith(f"{media_type}/") for media_type in media_types)
    
    if block_type == "text":
        text_content = block.get("text", "")
        return any(f"data:{media_type}" in text_content for media_type in media_types)
    
    if any(key in block for key in ["data", "base64_data"]):
        return block_type in ["image", "audio", "video", "file"]
    
    if "image_url" in block and "image" in media_types:
        image_url_obj = block["image_url"]
        return isinstance(image_url_obj, dict) and "url" in image_url_obj
    
    if "url" in block and block_type in ["image", "audio", "video"]:
        return True
    
    return False


async def get_media_descriptions_batch(vlm: BaseChatModel, media_blocks_info: List[Dict], chat_context: str = "") -> List[str]:
    logger = logging.getLogger(__name__)
    vlm_tasks = []
    
    for block_info in media_blocks_info:
        block = block_info['block']
        
        human_content = [block]
        if chat_context:
            human_content.insert(0, {"type": "text", "text": chat_context})
        
        messages = [
            SystemMessage(content=config.MEDIA_DESCRIPTION_PROMPT),
            HumanMessage(content=human_content)
        ]
        
        vlm_tasks.append(safe_vlm_invoke(vlm, messages))
    
    descriptions = await asyncio.gather(*vlm_tasks, return_exceptions=True)
    
    processed_descriptions = []
    error_count = 0
    
    for i, desc in enumerate(descriptions):
        if isinstance(desc, Exception):
            error_count += 1
            processed_descriptions.append(f"[Error processing {media_blocks_info[i]['media_type']}: {str(desc)}]")
        else:
            content = desc.content if hasattr(desc, 'content') else str(desc)
            processed_descriptions.append(content)
    
    if error_count > 0:
        logger.warning(f"Failed to process {error_count} out of {len(media_blocks_info)} media blocks")
    
    return processed_descriptions


async def safe_vlm_invoke(vlm: BaseChatModel, messages: List[BaseMessage]) -> Any:
    logger = logging.getLogger(__name__)
    try:
        return await vlm.ainvoke(messages)
    except Exception as e:
        logger.error(f"VLM invocation failed: {e}")
        raise e


async def replace_media_blocks_in_message(
    message: BaseMessage, 
    msg_idx: int, 
    media_blocks_info: List[Dict], 
    descriptions: List[str]
) -> BaseMessage:
    if not hasattr(message, 'content') or not isinstance(message.content, list):
        return message
    
    blocks_to_replace = {}
    for info, desc in zip(media_blocks_info, descriptions):
        if info['msg_idx'] == msg_idx:
            blocks_to_replace[info['block_idx']] = desc
    
    if not blocks_to_replace:
        return message
    
    new_content = []
    for block_idx, block in enumerate(message.content):
        if block_idx in blocks_to_replace:
            description = blocks_to_replace[block_idx]
            new_content.append({
                "type": "text",
                "text": f"[MEDIA DESCRIPTION: {description}]"
            })
        else:
            new_content.append(block)
    
    message_attrs = {}
    for attr in ['additional_kwargs', 'response_metadata', 'tool_calls', 'invalid_tool_calls', 'usage_metadata']:
        if hasattr(message, attr):
            message_attrs[attr] = getattr(message, attr)
    
    if isinstance(message, ToolMessage):
        message_attrs['tool_call_id'] = message.tool_call_id
    
    return type(message)(content=new_content, **message_attrs)


async def process_media_with_vlm(vlm: BaseChatModel, messages: List[BaseMessage], media_types: List[str] = None) -> List[BaseMessage]:
    if media_types is None:
        media_types = ["image", "audio", "video"]
    
    if not messages:
        return messages
    
    if not has_media(messages, media_types):
        return messages
    
    media_blocks_info = []
    for msg_idx, message in enumerate(messages):
        if not hasattr(message, 'content') or not isinstance(message.content, list):
            continue
            
        for block_idx, block in enumerate(message.content):
            if isinstance(block, dict) and is_media_block(block, media_types):
                media_blocks_info.append({
                    'msg_idx': msg_idx,
                    'block_idx': block_idx,
                    'block': block,
                    'media_type': block.get('type', 'unknown')
                })
    
    if not media_blocks_info:
        return messages
    
    chat_context = extract_chat_context(messages)
    descriptions = await get_media_descriptions_batch(vlm, media_blocks_info, chat_context)
    
    processed_messages = []
    for msg_idx, message in enumerate(messages):
        new_message = await replace_media_blocks_in_message(
            message, msg_idx, media_blocks_info, descriptions
        )
        processed_messages.append(new_message)
    
    return processed_messages


async def get_image_description(vlm: BaseChatModel, vlm_messages: List[Dict]) -> str:
    try:
        response = await vlm.ainvoke(vlm_messages)
        return response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        return f"[Error processing image: {str(e)}]"


def extract_chat_context(messages: List[BaseMessage]) -> str:
    text_blocks = []
    
    for message in messages:
        if hasattr(message, 'content'):
            if isinstance(message.content, str):
                text_blocks.append(message.content)
            elif isinstance(message.content, list):
                for block in message.content:
                    if isinstance(block, dict) and block.get('type') == 'text':
                        text_blocks.append(block.get('text', ''))
    
    return ' '.join(text_blocks)
