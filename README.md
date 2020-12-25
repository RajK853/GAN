# GAN

Personal repository to learn about different types of GAN models using Keras.

<p align="center">
    <img src="https://github.com/RajK853/GAN/blob/main/assets/mnist_gan.gif" width="640"\>
</p>

## Conda setup
1. Clone this repo.
  ```shell
  git clone https://github.com/RajK853/GAN.git $SRC_DIR
  ```
2. Create and activate conda environment.
  ```shell
  cd $SRC_DIR  
  conda env create -f environment.yml    
  conda activate gan-env
  ```
## Implementations

**Example:**

```shell
python train.py GAN ACGAN --epochs=100 --batch_size=128
```
The above command will train `GAN` and `ACGAN` models for 100 **epochs** with a **batch size** of 128.
> Use `--help` command to list out all the possible parameters.

### GAN
Implementation of normal *Generative Adversarial Network*.  

### ACGAN
Implementation of *Auxiliary Classifier Generative Adversarial Network*.  

### BiGAN
Implementation of normal *Bidirectional Generative Adversarial Network*.  

