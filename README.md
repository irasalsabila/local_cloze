# JavaCloze & SundaCloze

**Utilizing Synthetic Story Cloze Data for Commonsense Story Reasoning in Indonesian Local Languages**

---

This repository contains the code and datasets for the paper:

> **JavaCloze & SundaCloze: Utilizing Synthetic Story Cloze Data for Commonsense Story Reasoning in Indonesian Local Languages**

---

## Table of Contents

- [Introduction](#introduction)
- [Datasets](#datasets)
  - [Evaluation Datasets](#evaluation-datasets)
  - [Synthetic Training Datasets](#synthetic-training-datasets)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Training Models](#training-models)
  - [Evaluating Models](#evaluating-models)
- [Results](#results)
- [Contributors](#contributors)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Introduction

Story comprehension in NLP involves understanding complex causal and temporal relationships within narratives. While significant progress has been made in English, low-resource languages like Javanese and Sundanese lack sufficient datasets for such tasks due to the challenges and costs associated with manual data collection and annotation.

This project extends the Story Cloze Test framework to Javanese and Sundanese by:

- Creating evaluation datasets verified by native speakers.
- Generating synthetic training datasets using Large Language Models (LLMs).
- Evaluating machine learning models on these datasets to explore commonsense reasoning capabilities in low-resource languages.

---

## Datasets

### Evaluation Datasets

- **Size**: 500 manually verified samples for each language.
- **Format**: Each sample consists of a four-sentence premise and two possible endings (one correct, one incorrect).
- **Creation**: Translated and culturally adapted from the IndoCloze dataset, followed by a rigorous two-stage quality control process by native speakers.

### Synthetic Training Datasets

Generated using six different LLMs:

- **Closed-Weight Models**:
  - GPT-4o
  - Cohere Command R Plus
  - Claude 3 Opus
- **Open-Weight Models**:
  - Llama 3.1-70B
  - Gemma2
  - Mixtral

**Each dataset contains 1,000 samples per language.**

---

## Getting Started

### Prerequisites

- Python 3.7 or higher
- CUDA-enabled GPU (for training models)
- Internet connection (for accessing APIs of certain LLMs)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/JavaCloze-SundaCloze.git
   cd JavaCloze-SundaCloze
