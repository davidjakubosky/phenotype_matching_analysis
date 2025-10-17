from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple, Callable

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

try:
    from openai import AsyncOpenAI
except Exception as e:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore


class LlmError(Exception):
    pass


class LlmValidationError(LlmError):
    def __init__(self, obj: Dict[str, Any], raw_text: str, validation_error: str):
        super().__init__(f"Schema validation failed: {validation_error}")
        self.obj = obj
        self.raw_text = raw_text
        self.validation_error = validation_error


class LlmJSONClient:
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0, timeout: int = 60):
        if AsyncOpenAI is None:
            raise ImportError("openai package not available. Please install openai>=1.0.0")
        self.client = AsyncOpenAI(timeout=timeout)
        self.model = model
        self.temperature = temperature

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((LlmError,)),
    )
    async def create_json(
        self,
        messages: List[Dict[str, Any]],
        json_schema: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], str]:
        try:
            # Prefer JSON schema enforcement when supported
            response = await self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=messages,
                response_format=(
                    {"type": "json_schema", "json_schema": json_schema}
                    if json_schema is not None
                    else {"type": "json_object"}
                ),
            )
        except Exception as e1:  # type: ignore
            # Fallback without response_format
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=messages + [
                        {
                            "role": "system",
                            "content": "Return only a valid JSON object with double-quoted keys and strings.",
                        }
                    ],
                )
            except Exception as e2:
                raise LlmError(f"OpenAI request failed: {e2}")

        try:
            content = response.choices[0].message.content or "{}"
        except Exception as e:
            raise LlmError(f"No content in response: {e}")

        try:
            return json.loads(content), content
        except json.JSONDecodeError as e:
            raise LlmError(f"Invalid JSON returned: {e}: {content[:200]}")

    async def create_and_validate(
        self, 
        messages: List[Dict[str, Any]], 
        json_schema: Dict, 
        dataclass_factory: Callable[[Dict[str, Any]], Any]
    ) -> Tuple[Any, str]:
        """
        Create JSON response and validate it by constructing a dataclass.
        
        Args:
            messages: Chat messages
            json_schema: JSON schema for OpenAI structured output
            dataclass_factory: Function that creates a dataclass instance from a dict
                              (e.g., lambda d: LlmMappingResponse(**d))
        
        Returns:
            Tuple of (dataclass instance, raw response text)
        """
        obj, raw_text = await self.create_json(messages, json_schema)
        try:
            return dataclass_factory(obj), raw_text
        except (TypeError, ValueError, KeyError) as e:
            raise LlmValidationError(obj=obj, raw_text=raw_text, validation_error=str(e))
