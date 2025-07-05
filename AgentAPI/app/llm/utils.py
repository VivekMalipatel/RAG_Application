from typing import List, Dict, Any
import asyncio
import logging
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
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
    for message in messages:
        try:
            if message_contains_media(message, media_types):
                return True
        except Exception as e:
            logger.warning(f"Error checking message for media: {e}")
            continue
    
    return False


def message_contains_media(message: BaseMessage, media_types: List[str]) -> bool:
    if not isinstance(message, BaseMessage):
        raise ValueError(f"Expected BaseMessage, got {type(message)}")
    
    content = getattr(message, "content", None)
    
    if isinstance(content, str):
        for media_type in media_types:
            if f"data:{media_type}" in content:
                return True
    
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
                elif check_content_block_for_media(block, media_types):
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
        pseudo_blocks = []
        for key, value in additional_kwargs.items():
            if isinstance(value, str):
                pseudo_blocks.append({"type": "text", "text": value})
            elif isinstance(value, dict):
                pseudo_blocks.append(value)
        
        for block in pseudo_blocks:
            if isinstance(block, dict):
                if check_content_block_for_media(block, media_types):
                    return True
    
    return False


def check_content_block_for_media(block: dict, media_types: List[str]) -> bool:
    block_type = block.get("type")
    
    if block_type in media_types:
        return True
    
    if block_type == "text":
        text_content = block.get("text", "")
        for media_type in media_types:
            if f"data:{media_type}" in text_content:
                return True
    
    if block_type == "file":
        mime_type = block.get("mime_type", "")
        for media_type in media_types:
            if mime_type.startswith(f"{media_type}/"):
                return True
    
    if block_type in ["image_url", "audio_url", "video_url"]:
        media_type = block_type.split("_")[0]
        if media_type in media_types:
            return True
    
    if any(key in block for key in ["data", "base64_data", "url"]):
        if block_type in ["image", "audio", "video", "file"]:
            return True
    
    if "image_url" in block and "image" in media_types:
        image_url_obj = block["image_url"]
        if isinstance(image_url_obj, dict) and "url" in image_url_obj:
            return True
    
    return False


def is_media_block(block: dict, media_types: List[str]) -> bool:
    block_type = block.get('type', '')
    
    if block_type in media_types:
        return True
    
    if block_type in ['image_url', 'audio_url', 'video_url']:
        media_type = block_type.split('_')[0]
        return media_type in media_types
    
    if block_type == 'file':
        mime_type = block.get('mime_type', '')
        return any(mime_type.startswith(f"{media_type}/") for media_type in media_types)
    
    return False


def get_media_prompt(media_type: str) -> str:
    if media_type == "image":
        return config.IMAGE_DESCRIPTION_PROMPT
    else:
        return config.MEDIA_DESCRIPTION_PROMPT


async def get_media_descriptions_batch(vlm: BaseChatModel, media_blocks_info: List[Dict]) -> List[str]:
    logger = logging.getLogger(__name__)
    vlm_tasks = []
    
    for block_info in media_blocks_info:
        block = block_info['block']
        media_type = block_info['media_type']
        
        prompt = get_media_prompt(media_type)
        vlm_message = HumanMessage(content=[
            {"type": "text", "text": prompt},
            block
        ])
        
        vlm_tasks.append(safe_vlm_invoke(vlm, [vlm_message]))
    
    descriptions = await asyncio.gather(*vlm_tasks, return_exceptions=True)
    
    processed_descriptions = []
    for i, desc in enumerate(descriptions):
        if isinstance(desc, Exception):
            logger.error(f"Error processing media block {i}: {desc}")
            processed_descriptions.append(f"[Error processing {media_blocks_info[i]['media_type']}: {str(desc)}]")
        else:
            content = desc.content if hasattr(desc, 'content') else str(desc)
            processed_descriptions.append(content)
    
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
    
    relevant_blocks = [
        (info, desc) for info, desc in zip(media_blocks_info, descriptions)
        if info['msg_idx'] == msg_idx
    ]
    
    if not relevant_blocks:
        return message
    
    new_content = []
    blocks_to_replace = {info['block_idx']: desc for info, desc in relevant_blocks}
    
    for block_idx, block in enumerate(message.content):
        if block_idx in blocks_to_replace:
            description = blocks_to_replace[block_idx]
            new_content.append({
                "type": "text",
                "text": f"[MEDIA DESCRIPTION: {description}]"
            })
        else:
            new_content.append(block)
    
    message_class = type(message)
    message_kwargs = {
        'content': new_content
    }
    
    for attr in ['additional_kwargs', 'response_metadata', 'tool_calls', 'invalid_tool_calls', 'usage_metadata']:
        if hasattr(message, attr):
            message_kwargs[attr] = getattr(message, attr)
    
    if isinstance(message, ToolMessage):
        message_kwargs['tool_call_id'] = message.tool_call_id
    
    return message_class(**message_kwargs)


async def process_media_with_vlm(vlm: BaseChatModel, messages: List[BaseMessage], media_types: List[str] = None) -> List[BaseMessage]:
    if media_types is None:
        media_types = ["image", "audio", "video"]
    
    if not messages:
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
    
    descriptions = await get_media_descriptions_batch(vlm, media_blocks_info)
    
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
