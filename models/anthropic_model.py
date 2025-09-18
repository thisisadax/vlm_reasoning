import pandas as pd
from pathlib import Path
import re
from typing import Tuple
from .base_model import APIModel


class AnthropicModel(APIModel):
    """Handles inference for Anthropic's Claude models."""
    def __init__(self, api_model_identifier: str, **kwargs):
        self.api_model_identifier = api_model_identifier
        self.n_thinking_tokens = kwargs.get('n_thinking_tokens', 0)
        super().__init__(**kwargs)

    def _prepare_header(self) -> dict:
        return {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }

    def _prepare_endpoint(self, endpoint: str) -> str:
        return endpoint

    def build_vlm_payload(self, trial_metadata: pd.Series) -> dict:
        prompt_text = self.prompt
        image_paths = [
            Path(self.task.data_dir) / self.task.task_name / "trials" / f"trial={trial_metadata['trial_idx']}_{i}.png"
            for i in range(1, 7)
        ]
        encoded_images = [self._encode_image(p) for p in image_paths if p.exists()]
        
        content = [{"type": "text", "text": prompt_text}]
        content.extend([{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img}} for img in encoded_images])
        thinking_config = {"type": "enabled" if self.n_thinking_tokens >= 1024 else "disabled"}
        if thinking_config["type"] == "enabled":
            thinking_config["budget_tokens"] = self.n_thinking_tokens
        
        return {
            "model": self.api_model_identifier,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.max_tokens,
            "thinking": thinking_config
        }

    def _parse_response(self, response_json: dict) -> Tuple[str, str, dict]:
        thinking_text = [content['thinking'] for content in response_json['content'] if content['type']=='thinking'][0] if self.n_thinking_tokens >= 1024 else ''
        response_text = [content['text'] for content in response_json['content'] if content['type']=='text'][0]
        print(thinking_text)
        print(response_text)
        response_text = thinking_text + '\n' + response_text
        usage_metadata = response_json['usage']
        n_prompt_tokens = usage_metadata['input_tokens']
        n_output_tokens = usage_metadata['output_tokens']
        token_metadata = {'n_input_tokens': n_prompt_tokens, 'n_thought_tokens': n_output_tokens, 'n_output_tokens': n_output_tokens}
        matches = re.findall(r'\[([1-6])\]', response_text)
        if matches:
            answer = matches[-1]
        else:
            raise ValueError(f'No answer found in response: {response_text}')
        return response_text, answer, token_metadata