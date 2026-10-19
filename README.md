# SWMM-Agentic-Calibration
SWMM-Agentic-Calibration

# Agent-Based Calibration of Urban Drainage Models Using SWMM and LLMs
## Overview

This repository contains a Jupyter notebook demonstrating an agent-based approach to calibrating an urban drainage model (detention pond example). The workflow integrates the Storm Water Management Model (SWMM) with parameter optimization and large language model (LLM)-driven agents.


## Authors
This code has been created by Katharina Teuber at Jade Hochschule (University of Applied Sciences)
## Contents
.

├── sewer_calibration_agentic_generic_detention_pond-WST.ipynb

├── README.md

## Methodology

The workflow consists of the following steps:

### Model Setup
A SWMM model is loaded and configured programmatically.
Synthetic Observations
Simulation outputs are used to generate synthetic observation data for calibration.
Parameter Calibration
The SPOTPY library is used to perform parameter sampling and optimization based on an objective function.
Agent-Based Interaction
LLM-based agents interact with tools to:
run simulations
adjust model parameters
evaluate calibration performance
Requirements

### The following Python packages are required:

numpy
pandas
matplotlib
spotpy
swmm-api
langchain
langchain-core
langchain-ollama

A working SWMM installation is also required.

### Citation
** work in progress **, for appropriate citation, please contact the author.


### References and Thanks
Pichler, M. (2025). swmm_api: A Python Package for Automation, Customization, and Visualization in SWMM-Based Urban Drainage Modeling. Water, 17(9), 1373. https://doi.org/10.3390/w17091373
