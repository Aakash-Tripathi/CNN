# CNN
This repo contains a CNN implementation for the Linear and Matrix Algebra Course

## How to run

__Note:__ The code is not optimized and is not meant to be used in production, it is just a simple implementation of a CNN for the Linear and Matrix Algebra Course. 

1. Clone the repo
2. Install the requirements with `pip install -r requirements.txt`
3. Open the pytorch.ipynb file with jupyter notebook
4. Run the cells in order

__Note__: The code is written in Python 3.10.5 and was only trained using a GPU (using CUDA 11.7), so it is not guaranteed to work on a CPU.

## File Structure and Description

    ├── data                    # Contains the MNIST dataset
    ├── docs                    # Documentation of the project
    ├── models                  # Trained models
    │   ├── best_model.pt       # Best CNN model
    │   ├── model2.pt           # Best FCN model
    ├── src                     # Source code
    │   ├── lib                 # Library code for the project (CNN implementation)
    │   |   ├── data.py         # Code for loading the MNIST dataset and preprocessing it
    │   |   ├── model.py        # CNN model
    │   └── pytorch.ipynb       # Main file for the project
    ├── .gitignore              # Files to be ignored by git
    ├── README.md               # This file
    └── requirements.txt        # Requirements for the project
