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

## References

    [1] Saad Albawi, Tareq Abed Mohammed, and Saad Al-Zawi. Understanding of a convolutional neural network. In 2017 International Conference on Engineering and Technology (ICET), pages 1–6, 2017. 
    [2] Li Deng. The mnist database of handwritten digit images for machine learning research. IEEE Signal Processing Magazine, 29(6):141–142, 2012.
    [3] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Yoshua Bengio and Yann LeCun, editors,3rd International Conference on Learning Representations, ICLR2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings, 2015.
    [4] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. Imagenet classification with deep convolutional neural networks. In F. Pereira, C. J. C. Burges, L. Bottou, and K. Q. Weinberger, editors, Advances in Neural Information Processing Systems 25, pages 1097–1105. Curran Associates, Inc., 2012.
    [5] R.E. Uhrig. Introduction to artificial neural networks. In Proceedings of IECON ’95 - 21st Annual Conference on IEEE Industrial Electronics, volume 1, pages 33–37 vol.1, 1995.
