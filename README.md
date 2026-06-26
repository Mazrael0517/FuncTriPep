# FuncTriPep

FuncTriPep is a machine learning model based on a dual-branch feature fusion architecture, specifically designed for the high-throughput screening and identification of multifunctional bioactive peptides. By integrating advanced protein language models with traditional physicochemical properties, FuncTriPep overcomes the challenge of matrix interference in natural product isolation.

## 🌟 Core Features

* **Dual-Branch Feature Extraction**: Combines deep sequence semantic embeddings from the ESM-2 model with 30-dimensional handcrafted physicochemical features (including peptide length, hydrophobicity, charge, etc.).
* **Efficient Classifier**: Utilizes the LightGBM algorithm as the core prediction engine to achieve fast and accurate probability outputs.

## 📊 Model Performance

* **High Accuracy**: Achieved a classification accuracy of **95.13%** in the binary identification task for ACE-inhibitory (antihypertensive) peptides.
* **Robust Generalization**: On the independent test set, the model demonstrated outstanding performance with a Precision of **97.41%**, an F1-score of **0.9224**, and a ROC-AUC of **0.9919**, outperforming state-of-the-art methods.

## 🧪 Application & Validation

* **Virtual Screening**: Successfully applied to the high-throughput virtual screening of soybean protein ultrafiltration fractions (SPF, < 3 kDa).
* **In Vitro Validation**: Identified two novel and potent ACE-inhibitory peptides: 
    * **LARPSF** ($IC_{50} = 5.31 \ \mu\text{M}$)
    * **KLPPEAP** ($IC_{50} = 16.56 \ \mu\text{M}$)
* **In Vivo Efficacy**: A 5-week dietary intervention in Spontaneously Hypertensive Rats (SHR) proved that the peptide-enriched fraction significantly reduced blood pressure, alleviated cardiorenal damage, and improved gut microbiota composition.
