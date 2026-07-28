# 🧬 PMEST_model: Facet-Disentangled Metadata Vector Space Architecture

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![Manim](https://img.shields.io/badge/Manim-v0.18+-5c5c5c.svg)](https://www.manim.community/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> An end-to-end framework applying Dr. S. R. Ranganathan’s **PMEST** library classification theory to deep neural networks for multifaceted metadata extraction and disentangled vector sub-space retrieval.

---

## 📖 Table of Contents

- [Theoretical Foundation](#-theoretical-foundation)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Disentangled Vector Spaces vs. Single-Vector Baselines](#-disentangled-vector-spaces-vs-single-vector-baselines)
- [Repository Structure](#-repository-structure)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Quickstart](#-quickstart)
- [Generating Manim Architectural Diagrams](#-generating-manim-architectural-diagrams)
- [Citation & License](#-citation--license)

---

## 🔬 Theoretical Foundation

Traditional embedding models (e.g., Sentence-BERT, OpenAI text-embedding) map texts into a single monolithic vector space $E \in \mathbb{R}^d$. This leads to **attribute entanglement**, where dominant semantic topics suppress orthogonal attributes like publication year, methodology, or geographical origin.

**PMEST_model** overcomes this limitation by implementing library scientist Dr. S. R. Ranganathan's **PMEST Colon Classification Scheme**:

| Facet | Definition in PMEST_model | Example Metadata |
| :--- | :--- | :--- |
| **Personality ($P$)** | Core subject/domain, intrinsic entity, or methodology | *Graph Neural Networks, Transformers* |
| **Matter ($M$)** | Physical or conceptual materials, properties, or datasets | *CiteSeer, ImageNet, Text Corpora* |
| **Energy ($E$)** | Actions, processes, algorithms, operations, or tasks | *Classification, Node Embedding, Optimization* |
| **Space ($S$)** | Geographic locations, physical locations, or spatial regions | *Global, European Union, Urban Networks* |
| **Time ($T$)** | Chronological periods, timestamps, publication years | *2017, 2020, 2021, Contemporary Era* |

---

## ✨ Key Features

* **Dual-Channel Encoder:** Combines contextual language models (Transformer/BERT) with Graph Convolutional Networks (GCN) to process document text and citation graph structures simultaneously.
* **5-Head Facet Classifier:** Performs Span Boundary Extraction and Multi-Task Facet Classification across $P, M, E, S,$ and $T$.
* **Disentangled Vector Sub-Spaces:** Projects extracted metadata into distinct sub-spaces $V_P, V_M, V_E, V_S, V_T$.
* **Controllable Multi-Facet Retrieval:** Supports query-time dynamic weighting:
  $$\text{Sim}(Q, D) = \alpha S_P + \beta S_M + \gamma S_E + \delta S_S + \epsilon S_T$$
* **Production-Ready Visual Assets:** Executable Manim scripts to generate publication-grade vector space and GNN architectural graphics.

---

## 📐 System Architecture

The overall pipeline of PMEST_model flows through three major stages: