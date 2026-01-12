# Causal Analysis of Deception Mechanisms in Multi-Agent LLM Environments

**Authors:** Melih Kutay Yağdereli, Berk Kaan Elmas  
**Status:** Research Project (EE486 Statistical NLP)

## Overview
This repository contains the codebase for a controlled experimental framework designed to study **strategic deception** in Large Language Models (LLMs). Unlike standard hallucination detection, this project focuses on **deliberate, incentive-driven deception** in multi-agent settings.

We developed a custom **social deduction environment** (inspired by *Among Us*) where LLM agents (Llama 3) interact under conflicting objectives. The framework utilizes **structured communication protocols** to automatically verify claims against ground-truth state information, enabling scalable, human-free labeling of deceptive behavior.

## Key Features

### 1. Custom Social Deduction Environment
- A fully text-based, multi-agent game environment where agents (Impostors vs. Crewmates) must reason, communicate, and deceive to win.
- Implements partial observability and conflicting incentives to naturally induce strategic lying without explicit prompting.

### 2. Causal Analysis via Counterfactual Replay
- Introduces a **Counterfactual Replay Methodology** to measure the causal impact of specific deceptive acts.
- By intervening on specific deceptive statements and replaying the game simulation under identical conditions, we quantify how deception alters collective outcomes and belief convergence compared to truthful baselines.

### 3. Automated Deception Labeling
- Uses structured communication protocols to cross-reference agent statements with the environment's ground truth.
- Differentiates between **hallucinations** (unintentional errors) and **strategic deception** (intentional lies) by analyzing system entropy and context.

### 4. Credibility-Based Intervention
- Implements a dynamic **credibility scoring mechanism** that modulates an agent's influence based on verified truthfulness.
- Includes fine-tuned "Lie-Aware" models trained on interaction logs to minimize deceptive entropy and improve alignment.

## Tech Stack
- **Core:** Python
- **Models:** Llama 3 (via local inference or API)
- **Analysis:** Causal Inference, Entropy Analysis, Statistical NLP
- **Libraries:** PyTorch, Pandas, NumPy, Scikit-learn

## Methodology Highlights
- **Entropy Dynamics:** Analyzed how deceptive statements increase system entropy compared to truthful statements, establishing a metric for detection.
- **Intervention Success:** Demonstrated that credibility-aware incentive structures significantly reduce the rate of deception in autonomous agents.
