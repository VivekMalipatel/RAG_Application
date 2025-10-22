from typing import List, Dict, Any, Union, Optional
import os
import asyncio
import logging
import yaml
import json
from pathlib import Path
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_core.messages.content_blocks import is_data_content_block
from langchain import embeddings
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable
from openai import AsyncOpenAI
from langgraph.config import get_stream_writer


def load_media_description_prompt() -> str:
    llm_prompt_path = Path(__file__).parent / "prompt.yaml"
    try:
        with open(llm_prompt_path, 'r', encoding='utf-8') as f:
            prompt_data = yaml.safe_load(f)
            return prompt_data.get("MEDIA_DESCRIPTION_PROMPT")
    except (FileNotFoundError, yaml.YAMLError, KeyError):
        return "Provide an extremely detailed description of this media content. Include every visible/audible element, text, object, person, color, layout, sounds, speech, and any other relevant details without missing anything."


def _generate_media_key(block: dict) -> str:
    key_parts = []
    
    if block.get('type'):
        key_parts.append(f"type:{block['type']}")
    
    if 'url' in block:
        key_parts.append(f"url:{block['url']}")
    elif 'image_url' in block and isinstance(block['image_url'], dict):
        if 'url' in block['image_url']:
            key_parts.append(f"url:{block['image_url']['url']}")
    elif 'data' in block:
        key_parts.append(f"data:{str(block['data'])}")
    elif 'base64_data' in block:
        key_parts.append(f"base64_data:{str(block['base64_data'])}")
    
    if 'mime_type' in block:
        key_parts.append(f"mime:{block['mime_type']}")
    
    return "|".join(key_parts) if key_parts else str(sorted(block.items()))


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


async def get_media_descriptions_batch(vlm_client: AsyncOpenAI, model: str, media_blocks_info: List[Dict], chat_context: str = "", **model_kwargs) -> List[str]:
    logger = logging.getLogger(__name__)
    vlm_tasks = []
    
    for block_info in media_blocks_info:
        block = block_info['block']
        
        human_content = [block]
        if chat_context:
            human_content.insert(0, {"type": "text", "text": chat_context})
        
        messages = [
            {"role": "system", "content": load_media_description_prompt()},
            {"role": "user", "content": human_content}
        ]
        
        vlm_tasks.append(safe_vlm_invoke(vlm_client, model, messages, **model_kwargs))
    
    descriptions = await asyncio.gather(*vlm_tasks, return_exceptions=True)
    
    processed_descriptions = []
    error_count = 0
    
    for i, desc in enumerate(descriptions):
        if isinstance(desc, Exception):
            error_count += 1
            processed_descriptions.append(f"[Error processing {media_blocks_info[i]['media_type']}: {str(desc)}]")
        else:
            processed_descriptions.append(desc)
    
    if error_count > 0:
        logger.warning(f"Failed to process {error_count} out of {len(media_blocks_info)} media blocks")
    
    return processed_descriptions


async def safe_vlm_invoke(vlm_client: AsyncOpenAI, model: str, messages: List[Dict], **model_kwargs) -> str:
    logger = logging.getLogger(__name__)
    try:
        response = await vlm_client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            **model_kwargs
        )
        return response.choices[0].message.content
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


async def process_media_with_vlm(vlm_client: AsyncOpenAI, model: str, messages: List[BaseMessage], media_types: List[str] = None, **model_kwargs) -> List[BaseMessage]:
    if media_types is None:
        media_types = ["image", "audio", "video"]
    
    if not messages:
        return messages
    
    if not has_media(messages, media_types):
        return messages
    
    media_blocks_info = []
    seen_media_keys = set()
    unique_media_blocks = {}
    
    for msg_idx, message in enumerate(messages):
        if not hasattr(message, 'content') or not isinstance(message.content, list):
            continue
            
        for block_idx, block in enumerate(message.content):
            if isinstance(block, dict) and is_media_block(block, media_types):
                media_key = _generate_media_key(block)
                
                media_info = {
                    'msg_idx': msg_idx,
                    'block_idx': block_idx,
                    'block': block,
                    'media_type': block.get('type', 'unknown'),
                    'media_key': media_key
                }
                
                media_blocks_info.append(media_info)
                
                if media_key not in seen_media_keys:
                    seen_media_keys.add(media_key)
                    unique_media_blocks[media_key] = block
    
    if not unique_media_blocks:
        return messages
    
    unique_media_list = [
        {'block': block, 'media_type': block.get('type', 'unknown')}
        for block in unique_media_blocks.values()
    ]
    
    chat_context = extract_chat_context(messages)
    unique_descriptions = await get_media_descriptions_batch(vlm_client, model, unique_media_list, chat_context, **model_kwargs)
    
    media_key_to_description = {
        list(unique_media_blocks.keys())[i]: desc
        for i, desc in enumerate(unique_descriptions)
    }
    
    descriptions = [
        media_key_to_description[info['media_key']]
        for info in media_blocks_info
    ]
    
    processed_messages = []
    for msg_idx, message in enumerate(messages):
        new_message = await replace_media_blocks_in_message(
            message, msg_idx, media_blocks_info, descriptions
        )
        processed_messages.append(new_message)
    
    return processed_messages


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


def init_embeddings(model: str)-> Union[Embeddings, Runnable[Any, list[float]]]:
    from config import config
    return embeddings.init_embeddings(model=model, provider="openai", base_url=config.OPENAI_BASE_URL, api_key=config.OPENAI_API_KEY)


def is_message_sequence(payload: Any) -> bool:
    return isinstance(payload, list) and all(isinstance(item, BaseMessage) for item in payload)


def _announce_media_analysis(message: str = "Analysing Images.....\n\n") -> None:
    streaming_enabled = os.getenv("ENABLE_VLM_STREAM_UPDATES", "0") not in {"", "0", "false", "False"}
    if not streaming_enabled or not message:
        return
    try:
        writer = get_stream_writer()
        if writer:
            writer(message)
    except (RuntimeError, Exception):
        pass


def prepare_input_sync(
    payload: Any,
    vlm_client: AsyncOpenAI,
    vlm_model: str,
    vlm_model_kwargs: Dict[str, Any],
    *,
    logger: Optional[logging.Logger] = None,
    announcement: str = "Analysing Images.....\n\n",
) -> Any:
    if vlm_client is None or not vlm_model:
        return payload
    if not is_message_sequence(payload):
        return payload
    if not has_media(payload):
        return payload

    _announce_media_analysis(announcement)

    try:
        return asyncio.run(
            process_media_with_vlm(
                vlm_client,
                vlm_model,
                payload,
                **vlm_model_kwargs,
            )
        )
    except RuntimeError as exc:
        if "asyncio.run() cannot be called" in str(exc):
            if logger:
                logger.warning("Media preprocessing skipped because event loop is running")
            return payload
        raise


async def prepare_input_async(
    payload: Any,
    vlm_client: AsyncOpenAI,
    vlm_model: str,
    vlm_model_kwargs: Dict[str, Any],
    *,
    announcement: str = "Analysing Images.....\n\n",
) -> Any:
    if vlm_client is None or not vlm_model:
        return payload
    if not is_message_sequence(payload):
        return payload
    if not has_media(payload):
        return payload

    _announce_media_analysis(announcement)
    return await process_media_with_vlm(
        vlm_client,
        vlm_model,
        payload,
        **vlm_model_kwargs,
    )
