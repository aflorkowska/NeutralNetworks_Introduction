# :star: Getting started
This repo contains an explanation of the main terms associated with neural networks and consists of two parts. In the folder  "Theory", there are notebooks with the descriptions of the following topics, but the practical tasks and examples of use you can find in the folder "Exercises".

This repo was created as I wanted to improve my deep learning skills with the use of the PyTorch library and better understand the math behind these topics. I have relied on the PyTorch tutorial from this website: https://github.com/mrdbourke/pytorch-deep-learning/

### Content
Theory explanation:
1. Basics of NN.
2. RNN.
3. CNN.
4. Transfer learning and fine tuning.
5. Transformers, including vision transformers.
6. Graph vision NN.

Exercises:
1.
 
## :computer: Configuration - connection with GPU

In order to install PyTorch library I followed the instructions from this [tutorial](https://medium.com/@harunijaz/a-step-by-step-guide-to-installing-cuda-with-pytorch-in-conda-on-windows-verifying-via-console-9ba4cd5ccbef). If you want to get familiar with Tensorflow, I recommend to check this [tutorial](https://www.youtube.com/watch?v=5Ym-dOS9ssA&list=PLhhyoLH6IjfxVOdVC1P1L5z5azs0XjMsb&index=1) out.

1. Check compute capability for your GPU - [website](https://developer.nvidia.com/cuda-gpus).
2. Download and install specific driver for your GPU - [website](https://www.nvidia.com/Download/index.aspx).
3. Download and install Anaconda - [website](https://www.anaconda.com/download) - Do not forget to run .exe file as administrator.
4. Run as administrator (everytime!) `Anaconda Promt` window.
5. Create new enviroment for deep learning using command `conda create --name torch_gpu`
6. Download and install PyTroch using proper conda command [website](https://pytorch.org/).
7. Download and install CUDA from NVIDIA's [website](https://developer.nvidia.com/cuda-12-1-0-download-archive).
8. Download Cudnn from NVIDIA's [website](https://developer.nvidia.com/rdp/cudnn-archive) and extract Cudnn zip. file. Then select all, copy and paste it in the directory of CUDA.
6. Download and install PyCharm, Community Edition  - [website](https://www.jetbrains.com/pycharm/download/) - Do not forget to run .exe file as administrator.
7. Set proper Python interpreter.
```
Settings -> Python Interpreter -> Add Interpreter -> Add Local Interpreter-> Conda Enviroment -> Set path for the conda.exe file (in Scripts directory) -> Load environments-> Set suitable interpreter from the list (dedicated for tf env)
```
8. Check if pytorch was installed successfully
```
import torch
print(torch.cuda.is_available())

if torch.cuda.is_available():
    print("Cuda is Availabe")
else:
    print("Cuda Can't be found")
```

## :books: Recommended materials 
1. [Mathematical explanation of neutral networks, what they are, how they learn by 3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
2. [Lectures by Professor Adrian Horzyk, AGH University of Science and Technology, Cracow Poland](https://home.agh.edu.pl/~horzyk/lectures/kbcidmb/)
3. PyTorch tutorial: https://github.com/mrdbourke/pytorch-deep-learning/


Created by Agnieszka Florkowska, 2024