import pandas as pd
from pathlib import Path
from typing import Tuple
import re
from .base_model import APIModel


class GoogleModel(APIModel):
    """Handles inference for Google's Gemini models."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_thinking_tokens = kwargs.get('n_thinking_tokens', 0)

    def _prepare_header(self) -> dict:
        return {"Content-Type": "application/json"}

    def _prepare_endpoint(self, endpoint: str) -> str:
        return f"{endpoint}?key={self.api_key}"

    def build_vlm_payload(self, trial_metadata: pd.Series) -> dict:
        prompt_text = self.prompt
        image_paths = [
            Path(self.task.data_dir) / self.task.task_name / "trials" / f"trial={trial_metadata['trial_idx']}_{i}.png"
            for i in range(1, 7)
        ]
        encoded_images = [self._encode_image(p) for p in image_paths if p.exists()]
        
        parts = [{"text": prompt_text}]
        parts.extend([{"inline_data": {"mime_type": "image/png", "data": img}} for img in encoded_images])
        
        return {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "maxOutputTokens": self.max_tokens,
                "thinkingConfig": {
                    "thinkingBudget": self.n_thinking_tokens
                }
            }
        }

    def _parse_response(self, response_json: dict) -> Tuple[str, str, dict]:
        print(response_json)
        if 'candidates' not in response_json or not response_json['candidates']:
            return f"API Error: Content blocked or no candidates returned. Response: {response_json}"
        response_text = response_json['candidates'][0]['content']['parts'][0]['text']
        usage_metadata = response_json['usageMetadata']
        n_prompt_tokens = usage_metadata['promptTokenCount']
        n_thought_tokens = usage_metadata['thoughtsTokenCount'] if 'thoughtsTokenCount' in usage_metadata else 0
        token_metadata = {'n_input_tokens': n_prompt_tokens, 'n_thought_tokens': n_thought_tokens, 'n_output_tokens': n_thought_tokens}
        matches = re.findall(r'\[([1-6])\]', response_text)
        if matches:
            answer = matches[-1]
        else:
            raise ValueError(f'No answer found in response: {response_text}')
        return response_text, answer, token_metadata