# The Algorithmic Self-Portrait: Deconstructing Memory in ChatGPT

This repository contains the code accompanying our paper:

**_The Algorithmic Self-Portrait: Deconstructing Memory in ChatGPT_**,  
accepted at **The Web Conference 2026**.

The code enables reproduction of all analyses and figures reported in the paper.
The original dataset cannot be shared due to privacy considerations; AI-generated dummy data is provided, and the code supports custom user datasets.

---

## 📁 Repository Structure

The repository is organized by analysis section and corresponding figures in the paper.

### 1. Analysis of Theory-of-Mind Categories in ChatGPT Memories (Figure 1)
**Directory:** `/01_memories_and_mental_states`

- `analysis_tom.ipynb`  
  Annotates and analyzes Theory-of-Mind (ToM) categories in ChatGPT memories.

- `human_agreement.ipynb`  
  Evaluates inter-annotator agreement for ToM annotations.

---

### 2. Analysis of User-Initiated vs. Model-Initiated Memories (Figure 2)
**Directory:** `/02_user_agency`

- Code to analyze and visualize the relative agency of users and the model in memory creation.

---

### 3. Analysis of Personal and Special Category Data Types (Figure 3)
**Directory:** `/03_sensitive_information`

- Scripts and notebooks for identifying and categorizing sensitive and personal data types within memories.

---

### 4. Analysis of Provenance of Memories from User Messages (Figure 4)
**Directory:** `/04_provenance_of_memories`

- Analysis tracing how memories originate from user inputs and conversational context.

---

### 5. Reverse Engineering Memories: Training and Inference (Section 7)
**Directory:** `/05_reverse_engineering_memories`

- Code for training and evaluating models that reverse engineer memory construction mechanisms.

---

## 🔒 Data Availability

Due to **privacy considerations** and **IRB restrictions**, we are unable to release the original dataset used in the paper.

To support transparency and reproducibility:

- We provide **AI-generated dummy data** in the directory:  
  **`/dummy_data`**
- The dummy data mirrors the **structure, schema, and format** of the real dataset and allows all scripts and notebooks to run end-to-end.
- All plots in the paper can be reproduced using the dummy data, though numerical values may differ.

---
