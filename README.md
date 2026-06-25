# Nutrition Planner

An NLP-driven nutrition planner that turns a user's profile and goals into
personalised meal recommendations. It combines the **USDA FoodData Central**
nutrient database with an LLM to generate plans and meal variants, and can read
an InBody scan as input.

## Features

- Natural-language meal planning powered by an LLM (Groq / Hugging Face)
- Nutrient lookups backed by the USDA FoodData Central (SR Legacy) dataset
- Meal **variant generation** for variety within the same targets
- Jupyter notebooks documenting the full pipeline

## Tech Stack

- **Python**
- LLM APIs — **Groq** and **Hugging Face**
- pandas / data tooling over the FoodData Central CSVs
- Jupyter notebooks

## Project Files

```
app.py                    entry script
pipeline.py / .ipynb      core planning pipeline
variants.py / .ipynb      meal-variant generation
pipelineHuggingface.*     Hugging Face pipeline variant
FoodData_Central_*/       USDA nutrient dataset (CSV)
```

## Getting Started

```bash
pip install -r requirements.txt   # or install deps used in the notebooks
python app.py
```

## Configuration

API keys are read from environment variables — set your own before running:

```bash
export GROQ_API_KEY=your_key
export HF_TOKEN=your_token
```

> The code ships with placeholders (`<YOUR_GROQ_API_KEY>`, `<YOUR_HF_TOKEN>`).
> Never hardcode real keys — load them from the environment.
