# HW5: Deep neural network
## Hardware
* CPU: i7-12700K
* GPU: Nvidia GeForce RTX 3090
* RAM: 128G
## Environment
* OS: Ubuntu 22.04 LTS
* Anaconda: 4.10.3
* python: 3.9.12
* cudatoolKit in conda: 11.3.1
* pytorch: 1.11.0
* torchvision: 0.12.0  
P.S. The full requirement for anaconda is described in `package-list.txt`

## How to reproduce the work
First create conda environment and activate it.
```bash
conda create --name HW5 python=3.9.12
conda activate HW5
```
Install pytorch and and ipykernel.
```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -n HW5 ipykernel --update-deps --force-reinstall
```

Install others packages.
```bash
pip install -r requirements.txt
```

## Train the model
After create the environment, you can use `31055079_HW5.ipynb` to train the model.
Also, you can skip this step and directly use `inference.py` to evaluate the model performace.

## Evaluate
Use the following command to evaluate the model performance. It will read `HW5_weights.pth` to predict the result.
```bash
python inference.py
```