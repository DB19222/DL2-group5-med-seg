# Medical Multimodal Segmentation Using Foundation Models

### Danny van den Berg, Jesse Brouwers, Taiki Papandreou, and Roan van Blanken

---

This repository contains a reproduction and extension of ["SegVol: Universal and Interactive
Volumetric Medical Image Segmentation"](https://arxiv.org/abs/2311.13385) by Du et al. (2023). 

To read the full report containing detailed information on our reproduction experiments and extension study, please, refer to our [blogpost](blogpost.md).

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

To download the M3D-Seg dataset:

```data
python src/download_dataset.py
```

## Training and Inference 

## Evaluation and Demos

To perform evaluation of the results and reproduce our results, refer to the corresponding notebooks in the [`demos/`](demos/) folder.

## Results

### Reproduction

### Novel Contribution

---

## Snellius Compute Cluster Reproduction Instructions

In addition to having the ability to reproduce the results locally as described above, the repository contains a set of `.job` files stored in [`src/jobs/`](src/jobs) which have been used to run the code on the Senllius Compute Cluster. Naturally, if used elsewhere, these files must be adjusted to accommodate particular server requirements and compute access. In order to replicate the results in full, the following must be executed (in the specified order):

To retrieve the repository and move to the corresponding folder, run the following:

```repository
git clone git@github.com:DB19222/DL2-group5-med-seg.git
cd DL2-group5-med-seg/
```

To install the requirements, run the following:

```requirements
TODO
```
