import argparse
import re
import pandas as pd


SHAPE_NAMES_SINGULAR = {
    "3": "triangle",
    "4": "square",
    "5": "pentagon",
    "6": "hexagon",
    "7": "heptagon",
    "8": "octagon",
    "9": "nonagon",
}

SHAPE_NAMES_PLURAL = {k: v + "s" for k, v in SHAPE_NAMES_SINGULAR.items()}

NUMBER_WORDS_1_TO_10 = {
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
}


def preprocess_text(value):
    if value is None:
        return value
    if not isinstance(value, str):
        value = str(value)

    text = value
    text = text.replace(" s ", "s ")
    text = re.sub(r" s$", "s", text)

    text = re.sub(
        r"\b(3|4|5|6|7|8|9)\s+gons\b",
        lambda m: SHAPE_NAMES_PLURAL[m.group(1)],
        text,
    )
    text = re.sub(
        r"\b(3|4|5|6|7|8|9)\s+gon\b",
        lambda m: SHAPE_NAMES_SINGULAR[m.group(1)],
        text,
    )
    text = re.sub(
        r"\b(10|[1-9])\b",
        lambda m: NUMBER_WORDS_1_TO_10[m.group(1)],
        text,
    )

    # Remove periods and commas
    text = re.sub(r"[.,]", "", text)

    # Ensure lowercase
    text = text.lower()

    return text


# human_description is a list-like string; assume correct and eval then preprocess each element
def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess CSV text fields for model, human, and program descriptions.")
    parser.add_argument("input_csv", help="Path to input CSV file containing the data")
    parser.add_argument("output_csv", help="Path to output CSV file to write the processed data")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    # Columns to process
    program_col = "program_description"
    human_col = "human_description"
    model_col = "model_response"

    if program_col in df.columns:
        df[program_col] = df[program_col].apply(preprocess_text)

    if model_col in df.columns:
        df[model_col] = df[model_col].apply(preprocess_text)

    if human_col in df.columns:
        df[human_col] = df[human_col].apply(eval).apply(lambda lst: [preprocess_text(x) for x in lst])

    df.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()


