## :star: Getting started
This repo contains configuration instruction as well as the explanation of main terms associated with neural networks. README files (seperate for each topic) decribe the theory and the proper .py files concern the pratical part.

Content: basics of NN, CNN, RNN, Transfer learning and fine tuning, vision transformers, as well as graph vision NN. 
### :computer: Configuration - connection with GPU
There are some crucial steps you should follow to properly install tensorflow and use CUDA toolkit - [source](https://www.youtube.com/watch?v=5Ym-dOS9ssA&list=PLhhyoLH6IjfxVOdVC1P1L5z5azs0XjMsb&index=1).

1. Check compute capability for your GPU - [website](https://developer.nvidia.com/cuda-gpus).
2. Download and install specific driver for your GPU - [website](https://www.nvidia.com/Download/index.aspx).
3. Download and install Anaconda - [website](https://www.anaconda.com/download) - Do not forget to run .exe file as administrator.
4. Run as administrator (everytime!) `Anaconda Promt` window.
5. Create new enviroment for deep learning using command `conda create --name tf tensorflow-gpu`
6. Download and install PyCharm, Community Edition  - [website](https://www.jetbrains.com/pycharm/download/) - Do not forget to run .exe file as administrator.
7. Set proper Python interpreter.
```
Settings -> Python Interpreter -> Add Interpreter -> Add Local Interpreter-> Conda Enviroment -> Set path for the conda.exe file (in Scripts directory) -> Load environments-> Set suitable interpreter from the list (dedicated for tf env)
```
8. Check if tensorflow-gpu was installed successfully
```
import tensorflow as tf
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("GPUs Available: ", tf.config.list_physical_devices('GPU'))
```

## :books: Recommended materials 
1. [Mathematical explanation of neutral networks, what they are, how they learn by 3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
2. [Lectures by Professor Adrian Horzyk, AGH University of Science and Technology, Cracow Poland](https://home.agh.edu.pl/~horzyk/lectures/kbcidmb/)