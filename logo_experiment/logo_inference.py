
#from models.openai_logo_minimal import MinimalOpenAILogoModel
from anthropic_logo import MinimalAnthropicLogoModel  # fix the import
model = MinimalAnthropicLogoModel(
    api_file="/scratch/gpfs/nb0564/vlm_reasoning/api_metadata.json",  # use absolute path outside the experiment
    model_key="sonnet",
    prompt_file="/scratch/gpfs/nb0564/vlm_reasoning/logo_experiment/prompts/logo.txt",
    max_tokens=2048,
)

'''model = MinimalOpenAILogoModel(
    api_file="api_metadata.json",
    model_key="gpt-5",
    prompt_file="prompts/logo.txt",
    max_tokens=512,
)'''



df = model.infer_from_csv("/scratch/gpfs/nb0564/vlm_reasoning/logo_experiment/metadata.csv")

from preprocess import preprocess_text
# after df = model.infer_from_csv(...):
df["model_response"] = df["model_response"].apply(preprocess_text)
df.to_csv("output/logo/sonnet/logo.csv", index=False)
print(df.head())