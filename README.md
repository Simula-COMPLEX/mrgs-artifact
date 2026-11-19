
# MRGS: Metamorphic Relation-Guided Search

This repository contains the implementation and experimental artifacts for our research on search-based robustness testing of object detection software in robotic systems.

The approach uses metamorphic relations to guide multi-objective evolutionary search toward minimal image perturbations that expose robustness violations in deep learning-based object detectors.

## Use Cases

We evaluate six object detection models across three industrial robotic applications:

- **Face Detection:** YuNet face detection for human–robot interaction.
- **Sticker Detection:** Two SSD models for robotic laptop refurbishment.
- **Screw Detection:** Three YOLOv11 models for robotic laptop refurbishment.

## Search Algorithms

The experimental evaluation considers:

- NSGA-II
- SPEA2
- MOEA/D
- Random Search (baseline)

## Metamorphic Relations

The approach evaluates three properties:

- **MR1:** Detection preservation
- **MR2:** Confidence stability
- **MR3:** Localization stability

## Repository Structure

- `analysis/`: Scripts for analyzing experimental results for each research question.
- `use-cases/face-detection/`: Face detection experiments including algorithms, models and data.
- `use-cases/screw-detection/`: Screw detection experiments including algorithms, models and data.
- `use-cases/sticker-detection/`: Sticker detection experiments including algorithms, models and data.


## Installation

Clone the repository:

```bash
git clone https://github.com/Simula-COMPLEX/mrgs-artifact.git
cd mrgs-artifact
```

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```


**Note:** This repository uses Git LFS to store large model checkpoints. Please install Git LFS before cloning the repository to ensure that the trained models are downloaded correctly.


## Publication

This repository accompanies our research paper titled *Guiding Search-Based Robustness Testing with Metamorphic Relations*, currently under review. A link to the paper will be provided once it becomes available.