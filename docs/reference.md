# API Reference

Technical documentation for the `arxiv_recommender` core components. This reference is automatically generated from the source code docstrings using `mkdocstrings`.

---

## 🤖 LLM Engines
These classes handle communication with various LLM backends to analyze and rank arXiv papers.

### Local Inference
::: arxivrec.engine.llm.OLlamaLLM
    options:
        show_root_heading: true
        heading_level: 3
        show_source: true

### Cloud Inference
::: arxivrec.engine.llm.OpenaiLLM
    options:
        show_root_heading: true
        heading_level: 3
        show_source: true

---

## 📡 Data Retrieval & Processing
Components responsible for interfacing with external APIs and pre-processing raw metadata.



::: arxivrec.dataset.fetcher.ArxivFetcher
    options:
        show_root_heading: true
        heading_level: 3

---

## 🔔 Notification System
Modular classes to deliver the daily digest to different platforms.

| Notifier | Target Platform | Description |
| :--- | :--- | :--- |
| **Email** | SMTP / Inbox | Sends formatted HTML summaries. |
| **Slack** | Webhooks | Posts highlights to a dedicated channel. |
| **RSS** | XML Feed | Generates a local feed for readers. |

::: arxivrec.notify.notification.EmailNotifier
    options:
        show_root_heading: true
        heading_level: 4

::: arxivrec.notify.notification.SlackNotifier
    options:
        show_root_heading: true
        heading_level: 4

::: arxivrec.notify.notification.RSSNotifier
    options:
        show_root_heading: true
        heading_level: 4

---

## 🏗️ Core Pipeline
The central orchestrator that coordinates the retrieval, embedding, and ranking flow.

::: arxivrec.pipeline.LLMPipeline
    options:
        show_root_heading: true
        heading_level: 3
