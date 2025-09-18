import pandas as pd
from pathlib import Path
import re
from .base_model import APIModel


class OpenAIModel(APIModel):
    """Handles inference for OpenAI models."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reasoning_effort = kwargs.get('reasoning_effort', 'low')

    def _prepare_header(self) -> dict:
        #return {"Content-Type": "application/json", "api-key": self.api_key}
        return {"Content-Type": "application/json", 'Authorization': f'Bearer {self.api_key}'}

    def _prepare_endpoint(self, endpoint: str) -> str:
        return endpoint

    def build_vlm_payload(self, trial_metadata: pd.Series) -> dict:
        prompt_text = self.prompt
        image_paths = [
            Path(self.task.data_dir) / self.task.task_name / "trials" / f"trial={trial_metadata['trial_idx']}_{i}.png"
            for i in range(1, 7)
        ]
        encoded_images = [self._encode_image(p) for p in image_paths if p.exists()]
        
        content = [{"type": "input_text", "text": prompt_text}]
        content.extend([{"type": "input_image", "image_url": f"data:image/png;base64,{img}"} for img in encoded_images])
        
        return {"input": [{"role": "user", "content": content}], 
                "max_output_tokens": self.max_tokens, 
                "model": self.model_name,
                "reasoning": {"effort": self.reasoning_effort}}

    def _parse_response(self, response_json: dict) -> str:
        print(response_json)
        response_text = response_json['choices'][0]['message']['content']
        usage_metadata = response_json['usage']
        n_prompt_tokens = usage_metadata['prompt_tokens']
        n_output_tokens = usage_metadata['completion_tokens']
        n_thought_tokens = usage_metadata['completion_tokens_details']['reasoning_tokens']
        token_metadata = {'n_input_tokens': n_prompt_tokens, 'n_thought_tokens': n_thought_tokens, 'n_output_tokens': n_output_tokens}
        matches = re.findall(r'\[([1-6])\]', response_text)
        if matches:
            answer = matches[-1]
        else:
            raise ValueError(f'No answer found in response: {response_text}')
        return response_text, answer, token_metadata