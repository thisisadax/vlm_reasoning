import base64
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
from tqdm import tqdm
from anthropic import Anthropic


class MinimalAnthropicLogoModel:
    """Concise client for single-image prompts using Anthropic's Python SDK."""
    DEFAULT_MODELS = {
        "sonnet": "claude-sonnet-4-5-20250929",
        "opus": "claude-opus-4-1-20250805",
    }

    def __init__(
        self,
        api_file: str,
        model_key: str,
        prompt_file: str,
        api_model_identifier: str | None = None,
        max_tokens: int = 512,
        timeout: int = 60,
    ) -> None:
        config = json.loads(Path(api_file).read_text())
        creds = config[model_key]

        self.client = Anthropic(api_key=creds["api_key"])  # relies solely on SDK
        self.model = (
            api_model_identifier
            or creds.get("model")
            or self.DEFAULT_MODELS.get(model_key, "claude-3-5-sonnet-latest")
        )

        self.prompt: str = Path(prompt_file).read_text().strip()
        self.max_tokens = max_tokens

    @staticmethod
    def _encode_image(image_path: Path) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def _media_type(image_path: str) -> str:
        ext = Path(image_path).suffix.lower()
        if ext in {".jpg", ".jpeg"}:
            return "image/jpeg"
        elif ext == ".png":
            return "image/png"
        else: 
            raise ValueError(f"Unsupported image extension: {ext}")

    def _build_content(self, image_path: str) -> list[dict[str, Any]]:
        encoded_image = self._encode_image(Path(image_path))
        return [
            {"type": "text", "text": self.prompt},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": self._media_type(image_path),
                    "data": encoded_image,
                },
            },
        ]

    def _parse_response(self, response: Any) -> Tuple[str, Dict[str, int]]:
        texts: list[str] = []
        for block in getattr(response, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                texts.append(text)
        usage = getattr(response, "usage", None)
        token_meta = {
            "input_tokens": getattr(usage, "input_tokens", 0) if usage else 0,
            "output_tokens": getattr(usage, "output_tokens", 0) if usage else 0,
        }
        return "".join(texts), token_meta

    def infer_image(self, image_path: str) -> Tuple[str, Dict[str, int]]:
        content = self._build_content(image_path)
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        return self._parse_response(resp)

    def infer_from_csv(self, csv_path: str, image_column: str = "image_path") -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        responses: list[str] = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            text, _ = self.infer_image(str(row[image_column]))
            responses.append(text)

            # Clean, informative console print for prompt iteration
            img = str(row[image_column]).split("/")[-1]
            prog = str(row.get("program_description", "")).strip()
            # Truncate for neat logging
            def short(x, n=100):
                x = (x or "").replace("\n", " ").strip()
                return x if len(x) <= n else x[:n-1] + "…"

            print(f"[{img}]")
            print(f"  program: {short(prog)}")
            print(f"  model  : {short(text)}\n")

        df["model_response"] = responses
        return df

