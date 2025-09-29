import pandas as pd
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple
from PIL import Image, ImageDraw, ImageFont
import traceback
from tqdm import tqdm

from renderer.languages.base import BaseRenderer
from renderer.core import parse_program, render_strokes_to_image, export_image


class Task(ABC):
    """An abstract base class for generating task stimuli and oddball trials.

    This class provides a framework for defining a task, generating visual stimuli
    from program strings, and creating oddball-out trial sets based on abstract
    features defined in the stimulus metadata.
    """

    @property
    @abstractmethod
    def renderer(self) -> BaseRenderer:
        """An instance of a renderer class (e.g., DeterministicRenderer).
        
        This must be implemented by subclasses, typically as a class attribute.
        e.g., `renderer = DeterministicRenderer()`
        """
        pass

    def __init__(self, task_name: str, data_dir: str = "data", n_trials: int = 100, stroke_width: float = 3.0, **kwargs):
        """Initializes the Task instance, setting up paths and directories."""
        self.task_name = task_name
        self.data_dir = Path(data_dir)
        self.n_trials = n_trials
        self.stroke_width = stroke_width
        self._setup_paths()
        self._create_directories()

    def _setup_paths(self):
        """Initializes all necessary directory and file paths."""
        task_root = self.data_dir / self.task_name
        self.images_dir = task_root / "images"
        self.trials_dir = task_root / "trials"
        self.summary_dir = task_root / "summaries"
        self.metadata_path = task_root / "metadata.csv"
        self.trials_metadata_path = task_root / "trials.csv"

    def _create_directories(self):
        """Ensures that all required directories exist, creating them if necessary."""
        for path in [self.images_dir, self.trials_dir, self.summary_dir]:
            path.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def generate_programs(self) -> pd.DataFrame:
        """Generates program strings and their associated metadata."""
        pass

    # --- Main Orchestration ---
    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Executes the full pipeline. If trials.csv exists, load existing data.
        Otherwise, generate, render, and create all stimuli, oddball trials, and trial summaries.
        """
        # check if the final output file (trials.csv) already exists.
        if self.trials_metadata_path.exists():
            print(f"✅ Found existing trial data at '{self.trials_metadata_path}'. Skipping generation.")
            stimuli_df = pd.read_csv(self.metadata_path)
            trials_df = pd.read_csv(self.trials_metadata_path)
            return stimuli_df
        # if not, generate all stimuli, oddball trials, and example trials
        else:
            print("🔍 No existing trial data found. Starting full generation pipeline...")
            # 1. Generate and render all stimuli
            stimuli_df = self._generate_and_render_stimuli()
            # 2. Generate oddball trials from the stimuli
            trials_df = self._generate_all_oddball_trials(stimuli_df)
            # 3. Generate trial visualizations (visualize trial images and oddballs on a grid).
            self._generate_trial_summaries(trials_df)
            return stimuli_df, trials_df

    # --- Step 1: Stimulus Generation & Rendering ---
    def _generate_and_render_stimuli(self) -> pd.DataFrame:
        """Generates programs, renders them as images, and saves metadata."""
        # 1. Generate programs
        df = self.generate_programs()
        # 2. Render programs using the integrated renderer library
        filepaths = []
        for i, row in tqdm(df.iterrows(), desc="Rendering stimuli", unit="stimulus", leave=False, total=len(df)):
            output_path = self.images_dir / f"{i}.png"
            program_str = str(row['program_string'])
            try:
                ast = parse_program(program_str)
                strokes = self.renderer.evaluate(ast)
                image_array = render_strokes_to_image(strokes, line_width=self.stroke_width)
                export_image(image_array, str(output_path))
                #error
                filepaths.append(str(output_path))
            except Exception as e:
                #error
                print(f"❌ Error rendering program for row {i} ('{program_str[:50]}...'): {e}")
                print(traceback.format_exc())
                filepaths.append(None)
        
        # 3. Save metadata
        df["render_filepath"] = filepaths
        df.dropna(subset=['render_filepath'], inplace=True)
        df.to_csv(self.metadata_path, index=False)
        
        print(f"✅ Rendered {len(df)} stimuli for task '{self.task_name}'")
        print(f"✅ Images saved to: {self.images_dir}")
        print(f"✅ Metadata saved to: {self.metadata_path}")
        return df

    # --- Step 2: Oddball Trial Generation ---
    def _generate_all_oddball_trials(self, metadata_df: pd.DataFrame) -> pd.DataFrame:
        """Generates oddball trials for all specified abstractions."""
        print(f"\n🎯 Generating oddball trials for task '{self.task_name}'")
        excluded_cols = {'program_string', 'render_filepath'}
        abstraction_cols = [col for col in metadata_df.columns if col not in excluded_cols]
        print(f"📊 Found abstraction columns: {abstraction_cols}")
        all_trials_data = []
        for abstraction in abstraction_cols:
            pbar =  tqdm(range(self.n_trials), 
                         desc=f"🔄 Processing abstraction: {abstraction}", 
                         unit="trial",
                         leave=False)
            for _ in pbar:
                trial_data = self._create_single_trial(metadata_df, abstraction, abstraction_cols, len(all_trials_data))
                if trial_data:
                    all_trials_data.append(trial_data)
            print(f"✅ Generated {self.n_trials} trials for abstraction: {abstraction}")
        trials_df = pd.DataFrame(all_trials_data)
        if not trials_df.empty:
            trials_df.to_csv(self.trials_metadata_path, index=False)
            print(f"✅ Generated {len(trials_df)} oddball trials")
        return trials_df

    def _create_single_trial(self, df: pd.DataFrame, abstraction: str, all_abstractions: list, trial_idx: int) -> dict | None:
        """Attempts to create and save all assets for a single valid oddball trial."""
        try:
            target_value = random.choice(df[abstraction].unique())
            reference_stimuli = self._sample_reference_stimuli(df, abstraction, target_value, all_abstractions)
            oddball_stimulus = self._find_oddball_stimulus(df, reference_stimuli, abstraction, target_value)
            if oddball_stimulus is None or reference_stimuli is None: 
                return None
            return self._process_and_save_trial_assets(reference_stimuli, oddball_stimulus, abstraction, trial_idx)
        except Exception as e:
            print(f"❌ Error generating trial for {abstraction}: {e}")
            print(traceback.format_exc())
            return None

    def _sample_reference_stimuli(self, df: pd.DataFrame, abstraction: str, value: any, all_abstractions: list, max_attempts=int(5e4)) -> pd.DataFrame | None:
        """Samples 5 reference stimuli, ensuring no singletons on other dimensions."""
        matching_stimuli = df[df[abstraction] == value]
        if len(matching_stimuli) < 5:
            return None
        for _ in range(max_attempts):
            sample = matching_stimuli.sample(n=5)
            has_singleton = any(
                (sample[col].nunique() > 1 and (sample[col].value_counts() == 1).any())
                for col in all_abstractions if col != abstraction
            )
            if not has_singleton:
                return sample
        return None

    def _find_oddball_stimulus(self, df: pd.DataFrame, reference: pd.DataFrame, abstraction: str, target_value: any) -> pd.DataFrame | None:
        """Finds a valid oddball stimulus that differs only on the target abstraction."""
        allowed_values = {col: set(reference[col].unique()) for col in reference.columns if col not in {'program_string', 'render_filepath', abstraction}}
        mask = (df[abstraction] != target_value)
        for col, values in allowed_values.items():
            mask &= df[col].isin(values)
        candidates = df[mask]
        return candidates.sample(n=1) if not candidates.empty else None

    def _process_and_save_trial_assets(self, reference: pd.DataFrame, oddball: pd.DataFrame, abstraction: str, trial_idx: int) -> dict:
        """Combines stimuli, creates labeled images, and returns trial metadata."""
        trial_stimuli = pd.concat([reference, oddball])
        oddball_original_idx = oddball.index[0]
        shuffled_trial = trial_stimuli.sample(frac=1).reset_index()
        oddball_pos = shuffled_trial[shuffled_trial['index'] == oddball_original_idx].index[0] + 1
        for i, row in shuffled_trial.iterrows():
            labeled_path = self.trials_dir / f"trial={trial_idx}_{i+1}.png"
            self._add_label_to_image(row['render_filepath'], str(i + 1), str(labeled_path))
        return {
            'trial_idx': trial_idx,
            'oddball_idx': oddball_pos,
            'abstraction': abstraction,
            'stimuli_indices': shuffled_trial['index'].tolist()
        }

    # --- Step 3: Trial Visualization ---
    def _generate_trial_summaries(self, trials_df: pd.DataFrame):
        """Generates and saves 2x3 grid visualizations of example trials."""
        print("\n🖼️ Generating example trial visualizations")
        for abstraction in trials_df['abstraction'].unique():
            abstraction_trials = trials_df[trials_df['abstraction'] == abstraction]
            for i, (_, trial_row) in enumerate(abstraction_trials.iterrows()):
                self._create_trial_grid_image(trial_row, abstraction, i)
        print(f"✅ Trial visualizations saved to: {self.summary_dir}")

    def _create_trial_grid_image(self, trial_row: pd.Series, abstraction: str, example_idx: int):
        """Creates and saves a single 2x3 image grid for a given trial."""
        images = []
        for i in range(1, 7):
            image_path = self.trials_dir / f"trial={trial_row['trial_idx']}_{i}.png"
            if image_path.exists():
                img = Image.open(image_path).convert('RGB')
                border_color = "red" if i == trial_row['oddball_idx'] else "gray"
                images.append(self._add_border_to_image(img, border_color))
        if not images: 
            return
        img_w, img_h = images[0].size
        grid = Image.new("RGB", (3 * img_w, 2 * img_h), "white")
        for i, img in enumerate(images):
            grid.paste(img, ((i % 3) * img_w, (i // 3) * img_h))
        grid.save(self.summary_dir / f"{abstraction}_{example_idx}.png")

    # --- Static Utility Methods ---
    @staticmethod
    def _add_label_to_image(image_path: str, label: str, output_path: str):
        """Adds a red text label to the upper left corner of an image."""
        img = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 75)
        except IOError:
            font = ImageFont.load_default()
        draw.text((15, 15), label, fill="red", font=font)
        img.save(output_path)
    
    @staticmethod
    def _add_border_to_image(image: Image.Image, color: str, width: int = 10) -> Image.Image:
        """Adds a colored border to an image."""
        bordered_img = Image.new("RGB", (image.width + 2*width, image.height + 2*width), color)
        bordered_img.paste(image, (width, width))
        return bordered_img