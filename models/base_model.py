import json
import base64
import time
import requests
from typing import Tuple
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from abc import ABC, abstractmethod
from tenacity import retry, wait_exponential, stop_after_attempt

from tasks.base_task import Task


class APIModel(ABC):
    """
    An abstract base class for VLM inference models. It handles the core inference loop,
    data handling, and request logic, while delegating provider-specific details to subclasses.
    """
    def __init__(self, task: Task, 
                 model_name: str, api_file: str, 
                 max_tokens: int = 512, 
                 prompt_condition: str = None,
                 prompt_file: str = None, 
                 sleep: int = 0, 
                 cost_per_input_token: float = 0, 
                 cost_per_output_token: float = 0, 
                 **kwargs) -> None:
        self.task = task
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.prompt = open(prompt_file, 'r').read()
        self.sleep = sleep
        self.cost_per_input_token = cost_per_input_token
        self.cost_per_output_token = cost_per_output_token
        self.task_results_path = Path(f'output/{task.task_name}/{model_name}/{prompt_condition}.csv')
        self.task_results_path.parent.mkdir(parents=True, exist_ok=True) # ensure the output directory exists
        self.results_df = self.load_results() # Load the results DataFrame
        
        # Load API metadata from the specified file
        api_config = json.load(open(api_file, 'r'))
        if self.model_name not in api_config:
            raise KeyError(f"API metadata for '{self.model_name}' not found in {api_file}")
        
        self.api_key = api_config[self.model_name].get('api_key')
        print(f'api_key: {self.api_key}')
        raw_endpoint = api_config[self.model_name].get('endpoint')

        # Prepare endpoint and headers using subclass-specific logic
        self.endpoint = self._prepare_endpoint(raw_endpoint)
        self.header = self._prepare_header()
        print(f'header: {self.header}')

    @staticmethod
    def _encode_image(image_path: Path) -> str:
        """Encodes an image file to a base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    @abstractmethod
    def _prepare_header(self) -> dict:
        """Abstract method for preparing API request headers."""
        pass

    @abstractmethod
    def _prepare_endpoint(self, endpoint: str) -> str:
        """Abstract method for preparing the API endpoint URL."""
        pass

    @abstractmethod
    def build_vlm_payload(self, trial_metadata: pd.Series) -> dict:
        """Abstract method for constructing the provider-specific API payload."""
        pass

    @abstractmethod
    def _parse_response(self, response_json: dict) -> Tuple[str, str, dict]:
        """Abstract method for parsing the text from the API's JSON response."""
        pass
    
    #@retry(wait=wait_exponential(multiplier=1, min=1, max=20), stop=stop_after_attempt(10))
    def run_trial(self, task_payload: dict) -> Tuple[str, str, dict]:
        """Sends a request to the API and returns the parsed response."""
        response = requests.post(self.endpoint, headers=self.header, json=task_payload, timeout=400)
        if response.status_code != 200:
            raise requests.exceptions.HTTPError(f"API request failed with status {response.status_code}: {response.text}")
        response_json = response.json()
        if 'error' in response_json:
            raise ValueError(f"API returned an error: {response_json['error']}")
        response_text, answer, token_metadata = self._parse_response(response_json)
        return response_text, answer, token_metadata

    def load_results(self):
        """Loads the results DataFrame from a CSV file."""
        # First check if the results file exists
        if self.task_results_path.exists():
            results_df = pd.read_csv(self.task_results_path)
        # If not, load the trials metadata file and initialize the relevant results columns in the DataFrame.
        else:
            results_df = pd.read_csv(self.task.trials_metadata_path)
            results_df[['response', 'answer']] = None
            results_df[['n_input_tokens', 'n_thought_tokens', 'n_output_tokens']] = 0
        self.num_remaining_trials = results_df['response'].isna().sum()
        return results_df

    def run(self):
        """Main inference loop that processes all trials."""
        total_cost = self.calculate_cost()
        accuracy = self.calculate_accuracy()
        p_bar = tqdm(total=self.num_remaining_trials,
                     desc=f"🚀 Running {self.model_name}...",
                     postfix={'cost': f'${total_cost:.4f}', 'accuracy': f'{accuracy}%'})
        for i, trial in self.results_df.iterrows():
            # Check if trial has already been completed
            if pd.notna(trial.get('response')) and isinstance(trial.get('response'), str):
                continue
            # Construct the payload, run the trial, and save the results.
            payload = self.build_vlm_payload(trial)
            response, answer, token_metadata = self.run_trial(payload)
            # Update the results DataFrame.
            self.results_df.loc[i, 'response'] = response
            self.results_df.loc[i, 'answer'] = int(answer)
            self.results_df.loc[i, token_metadata.keys()] = token_metadata.values()
            print('response', response, '\n')
            print(f'oddball_idx: {trial.get('oddball_idx')} model_answer: {answer}\n')
            p_bar.set_postfix({'cost': f'${self.calculate_cost():.4f}', 'accuracy': f'{self.calculate_accuracy()}%'})
            p_bar.update(1)
            time.sleep(self.sleep)
            self.save_results()
        self.save_results()

    def save_results(self):
        """Saves the current results DataFrame to a CSV file."""
        self.results_df.to_csv(self.task_results_path, index=False)

    def calculate_cost(self) -> float:
        """Calculates the cost of the inference."""
        input_cost = self.results_df['n_input_tokens'].sum() / 1e6 * self.cost_per_input_token
        output_cost = self.results_df['n_output_tokens'].sum() / 1e6 * self.cost_per_output_token
        return input_cost + output_cost

    def calculate_accuracy(self) -> str:
        """Calculates model accuracy on the task."""
        pred_df = self.results_df[
            self.results_df.response.apply(lambda x: isinstance(x, str) and len(x) > 0)
        ]
        if pred_df.empty:
            return 'N/A'
        model_predictions = pred_df['answer']
        ground_truth = pred_df['oddball_idx']
        accuracy = int((model_predictions==ground_truth).mean() * 100)
        return str(accuracy)

        