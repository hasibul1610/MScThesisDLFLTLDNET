# DLFL-TLDNet: Tomato Leaf Disease Classification and Severity Estimation

DLFL-TLDNet is a hybrid framework that integrates deep learning and fuzzy logic for tomato leaf disease classification and severity estimation.

A pre-trained InceptionV3 model achieved **97.91% test accuracy** across ten tomato leaf disease categories. A fuzzy logic-based inference system estimates disease severity using lesion area and lesion count, categorizing severity into Primary, Medium, and Extreme levels.

The framework also incorporates Grad-CAM visualizations and color-coded overlays to support model interpretability. A Streamlit-based web application provides an interface for image-based disease diagnosis and severity reporting.

## Key Features

- Classification of 10 tomato leaf disease categories using InceptionV3.
- Fuzzy logic-based disease severity estimation.
- Lesion area and lesion count analysis.
- Grad-CAM visualization for model interpretability.
- Color-coded segmentation and overlay visualization.
- Streamlit-based web application.

## Repository Contents

| File | Description |
|---|---|
| `app4.py` | Streamlit application for disease classification and severity estimation |
| `inceptionv3-msc-thesis.ipynb` | InceptionV3 model training notebook |
| `inceptionv3_tomato.h5` | Trained InceptionV3 model weights (available in GitHub Releases) |
| `requirements.txt` | Required Python packages |

## Installation

Clone the repository:

```bash
git clone https://github.com/hasibul1610/MScThesisDLFLTLDNET.git
cd MScThesisDLFLTLDNET
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Download the Trained Model

Download `inceptionv3_tomato.h5` from the [GitHub Releases](https://github.com/hasibul1610/MScThesisDLFLTLDNET/releases) section.

Place the downloaded model file in the root directory of the repository, alongside `app4.py`.

## Run the Application

After downloading the trained model, run:

```bash
streamlit run app4.py
```

The application will open in your default web browser. Upload one or more tomato leaf images to obtain disease classification, severity estimation, segmentation results, and Grad-CAM visualizations.

## Dataset

The model was trained and evaluated using the tomato leaf image dataset described in the associated research manuscript.

The dataset is not included in this repository. Please refer to the original dataset source and the research manuscript for dataset details and access information.

## Research Paper

**Title:** Hybrid AI Model Integrating Deep Learning and Fuzzy Logic for Tomato Leaf Disease Detection and Severity Estimation (DLFL-TLDNet).

## License

The licensing terms for this repository will be specified separately.

## Contact

For questions regarding the implementation or research, please contact the corresponding author.
