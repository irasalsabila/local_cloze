# JavaCloze & SundaCloze: Story Cloze Test for Indonesian Local Languages

## Overview

This repository contains the source code and related assets for the paper "JavaCloze & SundaCloze: Utilizing Synthetic Story Cloze Data for Commonsense Story Reasoning in Indonesian Local Languages." The paper presents our work on expanding the Story Cloze Test to the Javanese and Sundanese languages, with datasets that help evaluate commonsense reasoning through narrative understanding. We employ synthetic data generation with the assistance of large language models (LLMs) to create training datasets and explore the application of commonsense reasoning in low-resource language contexts.

## Datasets

### Story Cloze Datasets
We constructed two main datasets for the Story Cloze Test in Javanese and Sundanese:

1. **Evaluation Dataset**: Consists of 500 samples that have undergone rigorous quality control by native speakers. This dataset is manually verified to ensure accuracy, fluency, and cultural relevance.

2. **Training Dataset**: Created synthetically using six different LLMs, including GPT-4o, Claude, Cohere Command R Plus, Mixtral, Gemma2, and Llama3.1. Each LLM generated 1,000 samples, providing six distinct training sets used in the model training process.

For details about dataset construction and quality assurance, refer to the paper's dataset section.

## Models

We experimented with several machine learning and transformer-based models to evaluate their performance on the Story Cloze Test in Javanese and Sundanese. These models include:

- **Pre-trained Language Models (PLMs)**: Fine-tuned Javanese BERT, Sundanese RoBERTa, and XLM-R.
- **Hierarchical BiLSTM**: A two-level BiLSTM model using FastText embeddings.
- **Similarity Models**: N-gram overlap-based classification and FastText similarity.

## Repository Structure

- **dataset/**: Contains the evaluation and training datasets in Javanese and Sundanese.
- **src/**: Code and configurations for the machine learning models.

## How to Use

### Prerequisites

- Python 3.8+
- Required Python Packages can be installed using `pip install -r requirements.txt` (it is recommended to use a virtual environment)
- [Fasttext Model for Javanese and Sundanese](https://fasttext.cc/docs/en/crawl-vectors.html)
- Access to LLM APIs (OpenAI, Cohere, Anthropic, etc.) if generating synthetic datasets

Jupyter notebooks in the **notebooks/** folder to see detailed data analysis and visualization of the model results.

## Acknowledgements

This work was conducted at Mohamed bin Zayed University of Artificial Intelligence (MBZUAI). Special thanks to our annotators for their contributions to ensuring the quality of our evaluation datasets.

## Contact

For any questions or collaboration inquiries, please contact:

- Salsabila Zahirah Pranida - salsabila.pranida@mbzuai.ac.ae
- Rifo Ahmad Genadi - rifo.genadi@mbzuai.ac.ae

