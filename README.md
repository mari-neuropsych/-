NeuralNetworkSimulations
Project Overview

This repository contains multiple Python simulations and analyses of neural networks, ranging from simple Hebbian learning models to advanced EEG-based cognitive modeling. The goal of these projects is to understand neural dynamics, decision-making processes, and the impact of stimuli on neural networks.

The repository includes:

Single-cell and Multi-cell Neural Simulations

Hebbian Learning Networks

Decision-making metrics: Reaction Time & Accuracy

EEG-based Neural Network Analysis

Contents & Features
1. Hebbian Learning Simulations

Simulated neural networks with 1, 10, 100, 1000 neurons.

Implemented Hebbian learning for multiple synthetic stimuli.

Visualized individual neuron outputs and average network response.

Explored network behavior and dynamics over time.

Achievements:

Developed understanding of Hebbian learning dynamics.

Visualized temporal evolution of neural networks.

Analyzed average network behavior across neurons.

2. Decision-Making & Reaction Time Simulations

Added threshold-based decision-making to each neuron.

Calculated reaction times and accuracy of decisions.

Compared performance of individual neurons vs multiple neurons.

Visualized outputs and decision metrics.

Achievements:

Implemented realistic neuron decision-making.

Extracted key performance indicators: RT & Accuracy.

Explored influence of network size on speed and accuracy.

3. Individual vs Combined Stimuli Analysis

Compared single stimuli vs combined stimuli responses.

Measured differences in reaction time and accuracy.

Created bar plots for visual comparison.

Achievements:

Analyzed impact of input complexity on network performance.

Visualized performance differences across scenarios.

Enhanced understanding of network integration of stimuli.

4. EEG-Based Neural Network Simulation

Downloaded real EEG data from OpenNeuro (sub-10, ses-1, rest task).

Applied preprocessing: normalization of signals.

Built a 1000-neuron network driven by EEG data.

Incorporated nonlinearity (tanh) and noise in neuron outputs.

Calculated Neural Strength, Decision Accuracy, Reaction Time.

Visualized results: neuron plots, heatmaps, cognitive indicators.

Achievements:

Connected neural network models to real EEG data.

Extracted cognitive metrics from brain signals.

Developed advanced visualization techniques for large networks.

Installation
# Clone repository
git clone https://github.com/<your_username>/NeuralNetworkSimulations.git
cd NeuralNetworkSimulations

# Install dependencies
pip install numpy matplotlib mne requests

Usage

Each Python script simulates a specific neural scenario:

hebbian_simulation.py: Hebbian learning on synthetic stimuli.

decision_simulation.py: Decision-making and reaction time analysis.

multi_stimuli_analysis.py: Individual vs combined stimuli comparison.

eeg_network_simulation.py: EEG-driven network simulation and cognitive metrics.

Run scripts individually to explore neuron outputs, network response, and decision metrics.

Visualizations

Line plots for individual neurons.

Average response across neurons.

Bar plots for reaction time and accuracy.

Heatmaps for neural activity across time.

Cognitive indicators derived from EEG data.

Achievements Summary

Built flexible neural network simulations.

Applied Hebbian learning and observed network dynamics.

Implemented decision-making metrics and cognitive indicators.

Analyzed performance under different stimulus conditions.

Integrated real EEG data to extract meaningful neural metrics.
