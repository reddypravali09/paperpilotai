# PaperPilot AI – Paper-to-Presentation & Viva Agent

**PaperPilot AI** is an intelligent, multi-agent academic assistant designed to simplify research paper analysis and presentation preparation. It transforms complex research papers into structured summaries, presentation slides, and potential viva questions with model answers.

Powered by Agentic AI, PaperPilot AI leverages specialized AI agents to understand research content, extract key concepts, generate presentations, and interact with users through an intelligent conversational assistant.

## Key Features

- **Research Paper Upload:** Upload and process research papers in PDF format.
- **AI-Powered Summarization:** Generate concise and structured summaries highlighting the key contributions, methodology, findings, and conclusions of a research paper.
- **Concept Extraction:** Identify and explain important technical concepts, terminology, and research-related information.
- **Automatic PPT Generation:** Convert research paper content into structured presentation slides, simplifying presentation preparation.
- **Viva Question Generation:** Generate potential examiner questions along with relevant model answers to help students prepare for research presentations and viva examinations.
- **Ask PaperPilot AI:** Interact with the AI assistant to ask questions about uploaded research papers, clarify concepts, and engage in general academic conversations.
- **Multi-Agent Architecture:** Employ multiple specialized AI agents coordinated through an intent router to handle different academic tasks efficiently.

## Agent Architecture

PaperPilot AI consists of specialized agents, each responsible for a specific task:

| Agent | Responsibility |
|---|---|
| Summarizer Agent | Generates concise research paper summaries |
| Concept Extractor Agent | Extracts and explains key concepts |
| Slide Generator Agent | Creates structured presentation content |
| Viva Agent | Generates potential viva questions and answers |
| General Agent | Handles user queries and conversations |
| Intent Router | Identifies user intent and routes requests to the appropriate agent |

## Technology Stack

- **Programming Language:** Python
- **Frontend:** Streamlit
- **PDF Processing:** Python PDF processing libraries
- **AI / NLP:** LLM-based processing and embeddings
- **Vector Storage:** Vector store for semantic retrieval
- **Presentation Generation:** Python-PPTX
- **Agent Architecture:** Modular multi-agent system

## Project Structure

```text
PaperPilotAI/
│
├── agents/
│   ├── base_agent.py
│   ├── concept_extractor.py
│   ├── general_agent.py
│   ├── intent_router.py
│   ├── slide_generator.py
│   ├── summarizer.py
│   └── viva_agent.py
│
├── config/
│   └── settings.py
│
├── core/
│   ├── embeddings.py
│   ├── pdf_processor.py
│   ├── text_chunker.py
│   └── vector_store.py
│
├── presentation/
│   └── pptx_builder.py
│
├── prompts/
├── schemas/
├── utils/
│
├── app.py
├── requirements.txt
└── .env.example
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/pravalicodes/paperpilotai.git
```

### 2. Navigate to the Project Directory

```bash
cd paperpilotai
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the virtual environment on Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

Create a `.env` file using the provided example:

```bash
copy .env.example .env
```

Add the required API keys and configuration values to your `.env` file.

### 6. Run the Application

```bash
streamlit run app.py
```

The application will be accessible through your browser at:

```text
http://localhost:8501
```

## Project Objective

The primary objective of PaperPilot AI is to make academic research more accessible and less time-consuming by automating the process of understanding research papers, preparing presentations, and anticipating viva questions.

By combining multiple specialized AI agents, the platform provides an integrated solution that helps students move from research paper analysis to presentation preparation and viva readiness within a single application.

## Future Enhancements

- Enhanced research paper question answering using Retrieval-Augmented Generation (RAG).
- Customizable presentation templates and slide designs.
- Advanced viva preparation with multiple difficulty levels.
- Voice-based interaction and responses.
- Support for multiple research papers and comparative analysis.

---

**PaperPilot AI – Transforming Research Papers into Presentation-Ready Insights.**
