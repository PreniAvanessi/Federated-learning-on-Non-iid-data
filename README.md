# Federated Learning under Non-IID Data Distributions

M.Sc. Data Science Project – University of Messina  
Author: Preni Avanessi  

## 📌 Overview
This project investigates the impact of **data heterogeneity** on federated learning systems.  
We evaluate three widely used algorithms:

- FedAvg
- FedProx
- FedNova

under different non-IID settings.

## 🎯 Objectives
- Analyze how **non-IID data** affects convergence and performance
- Compare algorithm robustness under:
  - Label skew
  - Quantity skew
- Study the impact of:
  - Number of clients (K)
  - Local epochs (E)
  - Heterogeneity level (α)

## 🧪 Experimental Setup
- Dataset: **CIFAR-10**, MNIST, Fashion-MNIST
- Model: Convolutional Neural Network (CNN)
- Metrics:
  - Accuracy
  - F1-score
  - AUC
  - Worst-client accuracy (fairness)

## 📊 Key Results
- **FedProx** is most robust under label skew
- **FedAvg** performs well under quantity skew
- **FedNova** shows instability under strong heterogeneity
- Label skew is significantly more challenging than quantity skew

## 📁 Repository Structure
