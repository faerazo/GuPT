# GuPT

GuPT is the name of the project developed by a student group for the course **Machine Learning for Natural Language Processing (DIT247)**. The system leverages extracted information from Gothenburg University’s (GU) bachelor’s and master’s courses (~590) and programs (~90), including relevant details from their websites and syllabus PDFs. This data is used as input to **GuPT**, which then employs a **Retrieval-Augmented Generation (RAG)** approach to respond to user queries. 

GuPT’s RAG model is built using **[LangChain](https://github.com/hwchase17/langchain)**, **OpenAI** embeddings, and **ChatGPT4o-mini**. By utilizing multi-querying and logic routing, GuPT can handle ambiguous questions and provide both specific and general answers regarding GU courses and programs. The goal is to offer a tool that efficiently provides information on entry requirements, learning objectives, and assessment methods, thereby reducing confusion and administrative workload.

---

## Table of Contents

1. [Features](#features)
2. [Getting Started](#getting-started)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Evaluation](#evaluation)
6. [Technologies Used](#technologies-used)
7. [Video Presentation](#video-presentation)

---

## Features

- **Natural Language Querying**: Ask questions about GU courses and programs in plain English.
- **Contextual RAG System**: Retrieves relevant information from a local database of course and program details.
- **Multi-Querying and Logic Routing**: Handles ambiguous queries and routes them through various queries to get precise answers.
- **Scalable**: Built to handle a large volume of course and program data.
- **Efficient Retrieval**: Reduces time spent searching for course or program information manually.

---

## Getting Started

These instructions will help you set up a local copy of GuPT for development and testing purposes.

### Prerequisites

- **Python 3.8+**: Ensure you have Python installed.  
- **pip**: Python package manager.  
- **OpenAI API Key**: Required for embedding and text generation. Obtain one from [OpenAI's website](https://platform.openai.com/).

---

## Installation

1. **Clone the Repository**

```bash
git clone https://github.com/faerazo/DIT247-NLP-Final-Project.git
cd DIT247-NLP-Final-Project
```

2. **Set Up Your `.env` File**

Create a file named `.env` in the project root and include your OpenAI API Key:

```bash
OPENAI_API_KEY=[YOUR_API_KEY]
```

3. **Install Required Libraries**

```bash
pip install -r requirements.txt
```

--- 

## Usage

Once you have the environment set up and the necessary dependencies installed, you can run GuPT and interact with the RAG Chatbot.

1. **Start the GuPT RAG Chatbot**

```bash
python rag.py
```

2. **Ask Your Questions**

Simply type your question or query into the chatbot interface or use one of the provided template questions.

---

## Evaluation

To evaluate GuPT’s responses on the test set (or a subset of it), run the following command:

```bash
python run_evaluation.py --subset 3
```
Where `--subset 3` indicates the subset of the test data you want to evaluate. Adjust this value as needed.

--- 

## Technologies Used
- [Firecrawl](https://www.firecrawl.dev)
- [JSONify](https://github.com/AustonianAI/pdf-to-json)
- [LangChain](https://github.com/hwchase17/langchain)
- [OpenAI](https://openai.com)
- [Chroma](https://docs.trychroma.com/docs/overview/introduction)
- [Gradio](https://gradio.app)

---

## Video Presentation

[![GuPT Video Presentation](https://img.youtube.com/vi/WVeGGnjLzEs/0.jpg)](https://youtu.be/WVeGGnjLzEs)
