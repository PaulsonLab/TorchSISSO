#  <p align="center">TorchSISSO: A PyTorch-Based Implementation of the Sure Independence Screening and Sparsifying Operator (SISSO) for Efficient and Interpretable Model Discovery

![torchsisso3](https://github.com/user-attachments/assets/a8d52ec3-3470-4807-904a-52525dc2b5d0)

## What is SISSO?

The **Sure Independence Screening and Sparsifying Operator (SISSO)** is a symbolic regression (SR) method that searches for interpretable models by exploring large feature spaces. It has proven particularly effective in fields like materials science, where it helps to uncover simple, accurate models that describe complex physical phenomena.

**TorchSISSO** is a native Python implementation of SISSO, built using the PyTorch framework. It addresses the limitations of the original FORTRAN-based SISSO ([SISSO]https://github.com/rouyang2017/SISSO) by providing a faster, more flexible, and easier-to-use solution.

### Key Features
- **GPU Acceleration**: TorchSISSO leverages PyTorch's GPU acceleration, offering significant speed-ups compared to the original SISSO implementation.
- **Seamless Integration**: Built in Python, TorchSISSO integrates easily with modern data science workflows, especially those using PyTorch and other Python-based libraries.
- **Enhanced Performance**: TorchSISSO has been shown to match or exceed the performance of the original SISSO across a range of scientific tasks, while dramatically reducing computation times.
- **Extensibility**: Designed with flexibility in mind, TorchSISSO can be easily extended to new problems and adapted for different types of data.

## Installation

You can install TorchSISSO via pip:
```
pip install TorchSisso
```

# Usage Examples
Usage of TorchSisso can be found in   <a href="https://colab.research.google.com/drive/1q0TEEALkb1PzJuusGKyHphv7tfod66XA?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
