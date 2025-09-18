import pandas as pd
from pathlib import Path
from .base_model import APIModel


class AzureModel(APIModel):
    """Handles inference for OpenAI models hosted on Azure."""

    def _prepare_header(self) -> dict:
        return {"Content-Type": "application/json", "api-key": self.api_key}

    def _prepare_endpoint(self, endpoint: str) -> str:
        return endpoint

    def build_vlm_payload(self, trial_metadata: pd.Series) -> dict:
        prompt_text = self.task.prompt
        image_paths = [
            Path(self.task.data_dir) / self.task.task_name / "trials" / f"trial={trial_metadata['trial_idx']}_{i}.png"
            for i in range(1, 7)
        ]
        encoded_images = [self._encode_image(p) for p in image_paths if p.exists()]
        
        content = [{"type": "text", "text": prompt_text}]
        content.extend([{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}} for img in encoded_images])
        
        return {"messages": [{"role": "user", "content": content}], "max_tokens": self.max_tokens}

    def _parse_response(self, response_json: dict) -> str:
        return response_json['choices'][0]['message']['content']