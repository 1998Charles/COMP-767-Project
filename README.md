# README

## Evaluate LLMs’ Code Semantic Understanding on Software Engineering Tasks: A Comparative Analysis

This repository contains:

1. **Preprocessed Subsets** of three major benchmark datasets:
   - **SubBigCloneBench** (for **Code Clone Detection**)
   - **SubCodeXGLUE** (for **Code Summarization**)
   - **SubMutantBench** (for **Equivalent Mutant Detection**)

2. **Jupyter Notebooks** demonstrating how to fine-tune and evaluate:
   - **General-Purpose LLMs**:  
     - BERT  
     - GPT-2  
     - T5  
   - **Code-Specific LLMs**:  
     - CodeBERT  
     - CodeGPT  
     - CodeT5  

These experiments align with the methodology and findings presented in our paper, focusing on code semantic understanding across the three tasks:  
1. **Code Summarization (c2s)**  
2. **Equivalent Mutant Detection (emd)**  
3. **Code Clone Detection (ccd)**

---

## Repository Structure

```plaintext
.
├── README.md                                   # This file
├── SubBigCloneBench_{train, validation, test}  # CCD dataset splits
│   ├── data-00000-of-00001.arrow
│   ├── dataset_info.json
│   └── state.json
├── SubCodeXGLUE_{train, validation, test}      # C2S dataset splits
│   ├── data-00000-of-00001.arrow
│   ├── dataset_info.json
│   └── state.json
├── SubMutantBench_{train, validation, test}    # EMD dataset splits
│   ├── data-00000-of-00001.arrow
│   ├── dataset_info.json
│   └── state.json
├── BertGPT2_c2s.ipynb
├── Bert_ccd.ipynb
├── Bert_emd.ipynb
├── GPT2_c2s.ipynb
├── GPT2_ccd.ipynb
├── GPT2_emd.ipynb
├── T5_c2s.ipynb
├── T5_ccd.ipynb
├── T5_emd.ipynb
├── codeBertGPT2_c2s.ipynb
├── codeBert_ccd.ipynb
├── codeBert_emd.ipynb
├── codeGPT_c2s.ipynb
├── codeGPT_ccd.ipynb
├── codeGPT_emd.ipynb
├── codeT5_c2s.ipynb
├── codeT5_ccd.ipynb
└── codeT5_emd.ipynb
```

### Datasets

Each dataset directory contains:
- One **.arrow** file that stores the preprocessed data  
- A **dataset_info.json** file with metadata  
- A **state.json** file with internal dataset states

**Train**, **Validation**, and **Test** splits are separate to facilitate straightforward evaluation.

### Notebooks

Each notebook is named to reflect:
- The **model** (e.g., `Bert`, `GPT2`, `T5`, `codeBert`, `codeGPT`, or `codeT5`)
- The **task** (e.g., `c2s` for code summarization, `ccd` for code clone detection, `emd` for equivalent mutant detection)

For example, `T5_c2s.ipynb` demonstrates code summarization experiments with T5.

---

## Requirements

To replicate our experiments and run the notebooks successfully, make sure you have the following:

1. **Python 3.10.15**  
2. **PyTorch 2.5.1** (GPU-enabled)  
3. **Transformers 4.45.2**  
4. **PEFT 0.14.0**  
5. **Datasets 3.2.0**  
6. **Evaluate 0.4.3**  
7. **Scikit-learn 1.6.0**  
8. **Jupyter Notebook** or **JupyterLab** to run `.ipynb` files  

Although you can install everything into an existing environment, we strongly recommend creating a dedicated virtual environment for reproducibility:

```bash
conda create -n code-lm python=3.10.15
conda activate code-lm
```

Then install the dependencies:

```bash
pip install torch==2.5.1
pip install transformers==4.45.2 peft==0.14.0 datasets==3.2.0 evaluate==0.4.3 scikit-learn==1.6.0
pip install jupyter
```

**Hardware**  
- Our experiments are conducted on an **NVIDIA L4 GPU**.  
- For large models (e.g., T5, GPT-2, CodeT5), a high-memory GPU is highly recommended to avoid out-of-memory errors.

---

## Usage Instructions

1. **Clone or Download this Repository**  
   ```bash
   git clone <REPO_URL>
   cd <REPO_NAME>
   ```

2. **Ensure the Preprocessed Datasets are in Place**  
   - If you are using this repository as is, the directories (e.g., `SubCodeXGLUE_train`) already contain the necessary files.  
   - If you have your own datasets, replace or link them in the corresponding `Sub<...>` folders with identical structure.

3. **Launch Jupyter Notebooks**  
   ```bash
   jupyter notebook
   ```
   Then open any of the available notebooks (e.g., `T5_c2s.ipynb`) in your browser.

4. **Run the Notebooks**  
   - **Load & Preprocess**: Each notebook starts by loading the respective dataset (train, validation, test) from the `.arrow` files.  
   - **Fine-tuning**: Follow the cells to fine-tune your chosen LLM on the corresponding task.  
   - **Evaluation & Metrics**: Each notebook computes task-specific metrics (e.g., BLEU for code summarization, F1-score for EMD and CCD).  

5. **Adjust Hyperparameters & Logging** (Optional)  
   - You can modify batch sizes, learning rates, and epochs directly within the notebooks.  
   - For reproducibility, we have included seed settings in most notebooks.

6. **Review Results**  
   - Results (loss curves, evaluation metrics, etc.) are displayed at the end of each notebook.  
   - Compare with our paper’s reported numbers or adapt the setup for your own experiments.

---

## Notes & Tips

- **Hardware Requirements**: Training large models like GPT-2 or T5 may require a GPU with sufficient memory. 
- **Data Splits**: The subsets here are intentionally smaller than full-scale benchmarks to facilitate faster prototyping. This may lead to slightly different metrics than those reported in large-scale experiments.  
- **Reproducing Paper Results**: Our paper’s main configurations are replicated in each notebook. For full dataset experiments, please consider substituting the subsets with the original benchmarks if you need the exact reported numbers.

---

## License and Citation

- This project is released under the [MIT License](LICENSE).  

---