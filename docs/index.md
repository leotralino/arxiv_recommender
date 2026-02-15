# ArXiv Recommender 🚀

**ArXiv Recommender** is a modular, AI-powered pipeline designed to help researchers keep up with the daily influx of new papers. It fetches daily papers, ranks them using LLMs, and notifies you via your preferred channel.

## Key Features
* **Modular Architecture**: Swap LLMs (OpenAI, Ollama, Gemini) and Notifiers (Email, Slack) via a registry system.
* **Vector-Based Ranking**: Uses state-of-the-art encoders to find papers that match your research interests.
* **CLI-First**: Designed for automation via GitHub Actions or local CRON jobs.

## System Architecture
```mermaid
graph TD
    subgraph Initialization
        Config[config.yaml] --> Loader[load_config]
        Loader --> Registries[Show Registries & Topics]
        Loader --> Objects[Init Encoder & Ranker]
    end

    subgraph TopicLoop ["Loop for each Topic"]
        Objects --> Fetcher[ArxivFetcher]
        Fetcher --> Pipeline[LLMPipeline]

        Pipeline --> Recommend[pipeline.recommend]
        subgraph InternalProcess ["Pipeline.recommend()"]
            Recommend --> F[Fetch Papers]
            F --> E[Encode & Similarity Search]
            E --> R[LLM Ranking]
        end

        InternalProcess --> Notify[pipeline.notify]
        Notify --> Channels[Email/Slack/RSS]
    end
```
