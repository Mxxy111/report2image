"""Async Client for NanoBanana API."""

import asyncio
import json
import logging
import aiohttp
import base64
import mimetypes
from pathlib import Path
from typing import Any, Dict, Optional

from config import AppConfig

logger = logging.getLogger(__name__)

def encode_image_to_data_uri(path_str: str) -> str:
    """
    将本地图片转换为 Data URI (Base64)。
    如果已经是 URL，则原样返回。
    """
    if path_str.startswith("http://") or path_str.startswith("https://"):
        return path_str
    
    path = Path(path_str)
    if not path.exists():
        logger.warning(f"Reference image not found: {path_str}. Treating as URL anyway.")
        return path_str
        
    mime_type, _ = mimetypes.guess_type(path)
    if not mime_type:
        mime_type = "image/png"
        
    with open(path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
    return f"data:{mime_type};base64,{encoded_string}"

class NanoBananaClient:
    def __init__(self, config: AppConfig):
        self.config = config
        self.headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def generate_image(
        self,
        prompt: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Call the API to generate image.
        Supports both Chat (v1/chat/completions) and Image (v1/images/generations) modes.
        """
        url = self.config.api_url
        mode = self.config.api_mode.lower()
        
        # 处理参考图 (支持本地路径转 Base64)
        ref_image = None
        if self.config.reference_image_url:
            ref_image = encode_image_to_data_uri(self.config.reference_image_url)
        
        payload = {}
        
        if mode == "image":
            # === 模式: v1/images/generations (OpenAI 变体) ===
            final_prompt = prompt
            if ref_image:
                # 注意：对于 Image 接口，直接拼接 Base64 字符串可能会导致 Prompt 过长而失败
                # 建议 Image 模式尽量使用 URL
                final_prompt = f"{prompt}\n{ref_image}"
            
            payload = {
                "model": self.config.model,
                "prompt": final_prompt,
                "n": self.config.num_images,
                "size": self.config.size, 
                "response_format": "url"
            }
            
        else:
            # === 模式: v1/chat/completions (Chat 接口) ===
            if not ref_image:
                # 纯文本
                user_content = prompt
            else:
                # 多模态 (User: Text + Image)
                user_content = [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": ref_image
                        }
                    }
                ]

            payload = {
                "model": self.config.model,
                "stream": False,
                "messages": [
                    {
                        "role": "user",
                        "content": user_content
                    }
                ]
            }
        
        # 合并额外参数
        if self.config.extra_params:
            payload.update(self.config.extra_params)

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    url, 
                    headers=self.headers, 
                    json=payload, 
                    timeout=self.config.rate_limit.timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"API Error [Status {response.status}]: {error_text}")
                        raise Exception(f"API Error {response.status}: {error_text}")
                    
                    return await response.json()
            except Exception as e:
                # Wrap exception for higher level handling
                logger.error(f"Request failed (URL: {url}): {type(e).__name__} - {e}")
                raise e
