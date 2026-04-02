
# 🧬 Generative AI Framework for Novel Drug Molecule Design and Optimization

## 👨‍💻 Project Batch – 12 Team Members

* TENNETY MAHESH DATTATREYA
* SINGURU LAHARI
* GARIKINA KARUNA
* JADA HARI SAI

---

# 📌 Project Description

This project presents a **Generative Artificial Intelligence framework** to generate **novel drug-like molecules** using deep learning models.

The system learns molecular patterns from chemical datasets and generates **chemically valid and optimized molecules**, which can assist in **drug discovery and pharmaceutical research**.

---

# 🔁 Overall Pipeline

SMILES → SELFIES → VAE → Transformer → GAN → Evaluation

### Explanation:

* **SMILES** → Molecular representation
* **SELFIES** → Robust encoding format
* **VAE** → Learns latent chemical features
* **Transformer** → Captures sequence dependencies
* **GAN** → Improves realism of molecules
* **Evaluation** → Filters drug-like candidates

---

# 📁 Project Structure

```
PROJECT/
│
├── SOURCE_CODE/
│   ├── main.py
│   └── model files
│
├── DOCUMENT/
│   └── Project_Report.pdf
│
├── PRESENTATION/
│   └── Final_PPT.pptx
│
├── RESULTS_SAMPLE/
│   ├── sample_output.txt
│   └── graphs.png
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# 🧠 Model Architecture

## 🔹 Variational Autoencoder (VAE)

* Encodes molecules into latent vectors
* Learns molecular distribution
* Generates new molecules via sampling

---

## 🔹 Transformer

* Uses self-attention
* Captures long-range dependencies
* Improves sequence modeling

---

## 🔹 Generative Adversarial Network (GAN)

* Generator → creates molecules
* Discriminator → validates molecules

---

# 📊 Evaluation Metrics

* **QED** → Drug-likeness
* **LogP** → Lipophilicity
* **Molecular Weight (MW)**

---

# 📦 Dataset

* Sample dataset included
* Full dataset not uploaded due to size limitations

---

# 📊 Sample Outputs

The project includes sample outputs in `RESULTS_SAMPLE/`:

* Generated molecules
* Training graphs
* Evaluation results

---

# ⚙️ System Requirements

* Python ≥ 3.8
* 8 GB RAM

### Recommended:

* GPU (Google Colab preferred)

⚠️ CPU execution is very slow.

---

# 📦 Installation

```
pip install -r requirements.txt
```

Install RDKit:

```
pip install rdkit-pypi
```

---

# 🚀 How to Run

## 🔹 Google Colab (Recommended)

1. Open Colab
2. Upload code
3. Enable GPU
4. Install requirements
5. Run all cells

---

## 🔹 Local System

```
pip install -r requirements.txt
python main.py
```

---

# ⏱️ Execution Time Notice

* Full training takes **2–3 hours**
* Depends on GPU availability

### Quick Test Mode:

* Use fewer epochs
* Use smaller dataset

---

# 🔄 Input / Output

### Input:

* SMILES dataset

### Output:

* Generated molecules with:

  * QED
  * LogP
  * Molecular Weight

---

# ⚠️ Important Notes

* RDKit is required
* Ensure dependencies are installed
* Verify file paths
* Use GPU for best performance

---

# 🎯 Conclusion

This project demonstrates how **VAE, Transformer, and GAN** models can be combined to generate **novel drug molecules**, highlighting the role of **Generative AI in drug discovery**.

---

# 📬 Final Remark

This project can be extended using:

* Larger datasets
* Advanced generative models
* Real-world validation

---
