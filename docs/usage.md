
# Usage Guide

## Running the Engine Locally
The simplest way to run the recommender is using the provided `Makefile`:

```bash
make run
```

## Configuration (`config.yaml`)

The engine is driven by a YAML configuration file. Here is a standard setup:

```yaml
topic:
  - id: "AI_Research"
    description: "Retrieval augmented generation (RAG), document parsing, table extraction"
    categories: ["cs.AI", "cs.LG"]

  - id: "IceCube_Particle_Astrophysics"
    description: "IceCube Neutrino Observatory, cosmic neutrinos, particle physics, dark matter searches, and high-energy astrophysical phenomena"
    categories: ["astro-ph.HE", "hep-ex", "astro-ph.CO"]

pipeline:
  simsearch_top_k: 10
  lookback_days: 1
  max_results: 100

models:
  encoder: "sentence-transformers/all-MiniLM-L6-v2"
  ranker:
    ollama:
      model_name: "llama3.2:3b"

notifiers:
  - email:
      host: "smtp.gmail.com"
      port: 465
```


### Command Line Arguments

If you need to specify a custom configuration file:

```bash
uv run python src/arxivrec/main.py --config path/to/your_config.yaml
```

!!! tip "Environment Variables"
Ensure you have `EMAIL_PASSWORD` or other variables set as your environment variables if you are using specific providers.
```
