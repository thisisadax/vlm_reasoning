import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import ast
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
from tqdm import tqdm

np.random.seed(1248)


# --- Configuration ---
CONFIG = {
    "model_name": 'Qwen/Qwen3-Embedding-8B',#'clip-ViT-B-32',
    "columns_to_use": ['program_description', 'human_description', 'model_response'],
    "output_csv": 'trial_level_comparison_results_v4.csv',
    "output_plot": 'trial_level_comparison_plot_v4.png'
}

class TrialLevelComparerV4:
    """
    Trial-level comparison of program vs. VLM descriptions to human descriptions
    without bootstrapping. Focuses on the significance of the similarity
    differences (VLM - Program) using a paired test.
    """
    def __init__(self, filepath: str, config: Dict[str, Any]):
        self.filepath = filepath
        self.config = config
        self.df = self._load_data()
        print("✅ Data loaded.")
        self.model = SentenceTransformer(self.config['model_name'], local_files_only=True)
        print(f"✅ Model '{self.config['model_name']}' loaded.")
        self.embedding_map = self._get_embeddings()
        print("✅ Embeddings generated for all unique sentences.")

    def _load_data(self) -> pd.DataFrame:
        """Loads and cleans the initial CSV file."""
        df = pd.read_csv(self.filepath)
        df = df[self.config['columns_to_use']].dropna().reset_index(drop=True)
        df['human_description'] = df['human_description'].apply(ast.literal_eval)
        return df

    def _get_embeddings(self) -> Dict[str, np.ndarray]:
        """Generates embeddings for all unique sentences."""
        all_human_descs = [desc for sublist in self.df['human_description'] for desc in sublist]
        unique_sentences = list(set(self.df['program_description']) | set(self.df['model_response']) | set(all_human_descs))
        embeddings = self.model.encode(unique_sentences, show_progress_bar=True)
        return {sentence: embedding for sentence, embedding in zip(unique_sentences, embeddings)}

    def _get_mean_similarity_to_group(self, source_sentence: str, target_sentences: List[str]) -> float:
        """Calculates the mean cosine similarity between a source sentence and a group of target sentences."""
        if not target_sentences: return 0.0
        source_embed = self.embedding_map[source_sentence]
        target_embeds = np.vstack([self.embedding_map[s] for s in target_sentences])
        
        # Normalize for cosine similarity calculation
        source_embed_norm = source_embed / np.linalg.norm(source_embed)
        target_embeds_norm = target_embeds / np.linalg.norm(target_embeds, axis=1, keepdims=True)
        
        similarities = target_embeds_norm @ source_embed_norm
        return float(np.mean(similarities))

    def _calculate_similarities(self):
        """Calculates matched similarities for each trial (no bootstrapping)."""
        print("Calculating trial-level similarity scores...")
        tqdm.pandas()
        # Calculate the matched similarity scores for each trial
        self.df['program_sim'] = self.df.progress_apply(
            lambda row: self._get_mean_similarity_to_group(row['program_description'], row['human_description']), axis=1
        )
        self.df['vlm_sim'] = self.df.progress_apply(
            lambda row: self._get_mean_similarity_to_group(row['model_response'], row['human_description']), axis=1
        )
        # Difference used for inference
        self.df['vlm_minus_program'] = self.df['vlm_sim'] - self.df['program_sim']
        print("✅ Matched similarity scores calculated.")

    def _generate_report(self):
        """Advisor-style report: matched vs shuffled null, above-null tests, per-trial p<.05, and paired test."""
        # Matched similarities (already computed in _calculate_similarities)
        prog = self.df['program_sim'].to_numpy()
        vlm  = self.df['vlm_sim'].to_numpy()
        n = len(self.df)

        # --- Build shuffled nulls by permuting the human_description lists across rows ---
        N_SHUFFLES = 200  # small but stable; bump to 1000 if you want tighter p-values
        human_lists = self.df['human_description'].to_list()

        prog_null_runs = []
        vlm_null_runs = []
        # Pre-grab the per-row program/model strings so we don't re-index in the loop
        progs = self.df['program_description'].tolist()
        vlms  = self.df['model_response'].tolist()

        for _ in range(N_SHUFFLES):
            perm = np.random.permutation(n)
            shuffled_humans = [human_lists[i] for i in perm]

            prog_null_runs.append(np.array([
                self._get_mean_similarity_to_group(progs[i], shuffled_humans[i]) for i in range(n)
            ]))
            vlm_null_runs.append(np.array([
                self._get_mean_similarity_to_group(vlms[i], shuffled_humans[i]) for i in range(n)
            ]))

        prog_null_runs = np.stack(prog_null_runs, axis=1)  # shape: (trials, shuffles)
        vlm_null_runs  = np.stack(vlm_null_runs,  axis=1)

        # Trial-wise null means
        prog_null_mean = prog_null_runs.mean(axis=1)
        vlm_null_mean  = vlm_null_runs.mean(axis=1)

        # Above-null per trial
        prog_above = prog - prog_null_mean
        vlm_above  = vlm  - vlm_null_mean

        # Per-trial one-sided p-values: P(null >= matched)
        prog_p = (1.0 + (prog_null_runs >= prog[:, None]).sum(axis=1)) / (N_SHUFFLES + 1.0)
        vlm_p  = (1.0 + (vlm_null_runs  >= vlm[:, None]).sum(axis=1)) / (N_SHUFFLES + 1.0)

        # Global tests
        t_prog, p_prog = stats.ttest_1samp(prog_above, 0.0)
        t_vlm,  p_vlm  = stats.ttest_1samp(vlm_above,  0.0)
        t_pair, p_pair = stats.ttest_rel(vlm, prog)

        vlm_wins = int((vlm > prog).sum())

        # Pretty print in advisor’s format
        report = f"""
{'='*70}
TRIAL-LEVEL SIMILARITY ANALYSIS
{'='*70}
Mean similarity to matched human descriptions:
Program: {prog.mean():.4f}
VLM:     {vlm.mean():.4f}

Mean similarity to shuffled (null) human descriptions:
Program: {prog_null_mean.mean():.4f}
VLM:     {vlm_null_mean.mean():.4f}

Mean above-null performance (matched - shuffled):
Program: {prog_above.mean():.4f}  (t={t_prog:.2f}, p={p_prog:.4f})
VLM:     {vlm_above.mean():.4f}  (t={t_vlm:.2f}, p={p_vlm:.4f})

Trials significantly above null (p<0.05):
Program: {(prog_p < 0.05).sum()}/{n} ({(prog_p < 0.05).mean()*100:.1f}%)
VLM:     {(vlm_p  < 0.05).sum()}/{n} ({(vlm_p  < 0.05).mean()*100:.1f}%)

Direct comparison (VLM vs Program at trial level):
Mean difference (VLM - Program): {(vlm - prog).mean():.4f}
Paired t-test: t={t_pair:.2f}, p={p_pair:.4f}
VLM better on {vlm_wins}/{n} trials ({vlm_wins/n*100:.1f}%)
{'='*70}
        """
        print(report.strip())


    def _plot_results(self):
        """Generates and saves plots focused on direct trial-level comparison."""
        print("Generating plots...")
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        title_style = {'fontsize': 15, 'fontweight': 'bold', 'pad': 12}
        label_style = {'fontsize': 12, 'fontweight': 'bold'}
        
        # Plot 1: Scatter plot for direct comparison
        ax = axes[0]
        ax.scatter(self.df['program_sim'], self.df['vlm_sim'], alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, 'k--', alpha=0.6, label='y=x (Equal Similarity)')
        ax.set_xlabel('Program Similarity to Human', **label_style)
        ax.set_ylabel('VLM Similarity to Human', **label_style)
        ax.set_title('Direct Comparison: VLM vs. Program', **title_style)
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Plot 2: Histogram of the differences
        ax = axes[1]
        diff = self.df['vlm_minus_program']
        sns.histplot(diff, bins=30, kde=True, ax=ax, color='#9b59b6')
        ax.axvline(0, color='k', linestyle='--', label='No Difference')
        ax.axvline(diff.mean(), color='r', linestyle='-', label=f'Mean Diff = {diff.mean():.3f}')
        ax.set_xlabel('Similarity Difference (VLM - Program)', **label_style)
        ax.set_ylabel('Number of Trials', **label_style)
        ax.set_title('Distribution of Similarity Differences', **title_style)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.config['output_plot'], dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to '{self.config['output_plot']}'")
        plt.close()

    def run_analysis(self):
        """Runs the full trial-level comparison pipeline (no bootstrapping)."""
        self._calculate_similarities()
        self._generate_report()
        self._plot_results()
        
        self.df.to_csv(self.config['output_csv'], index=False)
        print(f"✅ Full results saved to '{self.config['output_csv']}'")
        return self.df

if __name__ == '__main__':
    analyzer = TrialLevelComparerV4(
        filepath="/scratch/gpfs/nb0564/vlm_reasoning/logo_experiment/output/logo/sonnet/logo_processed.csv",
        config=CONFIG
    )
    results_df = analyzer.run_analysis()

