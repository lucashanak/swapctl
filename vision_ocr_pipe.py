"""
title: Vision OCR Pipe
author: lucas
version: 1.0.0
description: Routes images through a vision/OCR model, then streams the main model response. Fully async.
"""

import json
from typing import Any, Awaitable, Callable, Optional

import aiohttp
from pydantic import BaseModel, Field


class Pipe:
    class Valves(BaseModel):
        LLAMA_SWAP_URL: str = Field(
            default="http://192.168.1.94:8080",
            description="llama-swap base URL",
        )
        VISION_MODEL: str = Field(
            default="qwen3-vl-8b",
            description="Vision model ID in llama-swap for OCR",
        )
        MAIN_MODEL: str = Field(
            default="qwen3.6-35b-a3b",
            description="Main reasoning model ID in llama-swap",
        )
        OCR_PROMPT: str = Field(
            default="Extract ALL text from this image accurately. Preserve formatting, tables, and structure. Output the text exactly as it appears.",
            description="System prompt for OCR extraction",
        )

    def __init__(self):
        self.valves = self.Valves()

    def pipes(self):
        return [
            {
                "id": "vision-ocr",
                "name": f"Vision OCR ({self.valves.VISION_MODEL} + {self.valves.MAIN_MODEL})",
            }
        ]

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
        __files__: Optional[list] = None,
    ):
        try:
            messages = body.get("messages", [])
            if not messages:
                yield "No messages provided."
                return

            images = self._extract_images(messages)

            if not images:
                async for chunk in self._stream_model(self.valves.MAIN_MODEL, messages):
                    yield chunk
                return

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Reading {len(images)} image(s) with {self.valves.VISION_MODEL}...",
                            "done": False,
                        },
                    }
                )

            ocr_results = []
            for img_data in images:
                ocr_text = await self._call_vision_model(img_data)
                ocr_results.append(ocr_text)

            combined_ocr = "\n\n---\n\n".join(ocr_results)

            if not combined_ocr.strip() or combined_ocr.strip() in ("-", "—", "."):
                yield f"**OCR Warning:** {self.valves.VISION_MODEL} returned no useful text from the image. Try switching to qwen3-vl-8b in Valves.\n\n"

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Analyzing with {self.valves.MAIN_MODEL}...",
                            "done": False,
                        },
                    }
                )

            text_messages = self._build_main_messages(messages, combined_ocr)

            async for chunk in self._stream_model(self.valves.MAIN_MODEL, text_messages):
                yield chunk

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "", "done": True},
                    }
                )

        except Exception as e:
            yield f"\n\n**Pipe Error:** {type(e).__name__}: {e}"

    def _extract_images(self, messages: list) -> list:
        images = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        if url.startswith("data:image"):
                            images.append(url)
        return images

    async def _call_vision_model(self, image_data_url: str) -> str:
        payload = {
            "model": self.valves.VISION_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                        {"type": "text", "text": self.valves.OCR_PROMPT},
                    ],
                }
            ],
            "max_tokens": 4096,
            "temperature": 0.1,
            "stream": False,
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.valves.LLAMA_SWAP_URL}/v1/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        return f"[OCR Error: HTTP {resp.status} - {body[:200]}]"
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"[OCR Error: {type(e).__name__}: {e}]"

    async def _stream_model(self, model_id: str, messages: list):
        payload = {
            "model": model_id,
            "messages": messages,
            "stream": True,
            "max_tokens": 4096,
            "temperature": 0.1,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        try:
            timeout = aiohttp.ClientTimeout(total=600, sock_read=300)
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.valves.LLAMA_SWAP_URL}/v1/chat/completions",
                    json=payload,
                    timeout=timeout,
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        yield f"\n\n[Stream Error: HTTP {resp.status} - {body[:200]}]"
                        return
                    buffer = ""
                    async for raw_chunk in resp.content:
                        buffer += raw_chunk.decode("utf-8", errors="replace")
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()
                            if not line or not line.startswith("data: "):
                                continue
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                return
                            try:
                                chunk = json.loads(data_str)
                                delta = chunk["choices"][0].get("delta", {})
                                content = delta.get("content") or ""
                                if content:
                                    yield content
                                else:
                                    reasoning = delta.get("reasoning_content") or ""
                                    if reasoning:
                                        yield reasoning
                            except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                                continue
        except Exception as e:
            yield f"\n\n[Stream Error: {type(e).__name__}: {e}]"

    def _build_main_messages(self, original_messages: list, ocr_text: str) -> list:
        result = [
            {
                "role": "system",
                "content": f"The user sent an image. Here is the text extracted from it via OCR:\n\n```\n{ocr_text}\n```\n\nUse this extracted text to answer the user's question.",
            }
        ]
        for msg in original_messages:
            content = msg.get("content")
            if isinstance(content, list):
                text_parts = [
                    p.get("text", "")
                    for p in content
                    if p.get("type") == "text"
                ]
                text = "\n".join(text_parts).strip()
                if text:
                    result.append({"role": msg["role"], "content": text})
            elif isinstance(content, str):
                result.append({"role": msg["role"], "content": content})
        return result
