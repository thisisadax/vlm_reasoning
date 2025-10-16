import pandas as pd
from pathlib import Path
from typing import Tuple
import re
from .base_model import APIModel


class QwenModel(APIModel):
    """Handles inference for Qwen models via Together.xyz API."""

    def __init__(self, max_tokens: int = 512, **kwargs):
        self.max_tokens = max_tokens
        super().__init__(**kwargs)

    def _prepare_header(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _prepare_endpoint(self, endpoint: str) -> str:
        return endpoint

    def build_vlm_payload(self, trial_metadata: pd.Series) -> dict:
        """Build payload for Qwen model via Together.xyz API."""
        prompt_text = self.prompt

        # Get image paths
        image_paths = [
            Path(self.task.data_dir) / self.task.task_root_name / "trials" / f"trial={trial_metadata['trial_idx']}_{i}.png"
            for i in range(1, 7)
        ]
        encoded_images = [self._encode_image(p) for p in image_paths if p.exists()]

        # Build content with text and images
        content = [{"type": "text", "text": prompt_text}]
        content.extend([{
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img}"}
        } for img in encoded_images])

        return {
            "model": "liuhaotian/llava-v1.5-7b",  # Free LLaVA model on Together.xyz
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.max_tokens,
            "temperature": 0.1
        }

    def _parse_response(self, response_json: dict) -> Tuple[str, str, dict]:
        """Parse response from Together.xyz API."""
        if 'choices' not in response_json or not response_json['choices']:
            raise ValueError(f"Invalid response format: {response_json}")

        response_text = response_json['choices'][0]['message']['content']

        # Extract answer from brackets like [3]
        matches = re.findall(r'\[([1-6])\]', response_text)
        if matches:
            answer = matches[-1]
        else:
            raise ValueError(f'No answer found in response: {response_text}')

        # Extract token usage
        usage = response_json.get('usage', {})
        token_metadata = {
            'n_input_tokens': usage.get('prompt_tokens', 0),
            'n_thought_tokens': 0,  # Qwen doesn't have thinking tokens
            'n_output_tokens': usage.get('completion_tokens', 0)
        }

        return response_text, answer, token_metadata
