# GAN
Personal repository to learn about different types of GAN models using Keras.

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

```shell
python train.py --model=$MODEL_TYPE
```
Here `$MODEL_TYPE` is the name of one of the given GAN models.

### GAN
Implementation of normal *Generative Adversarial Network*.  

### ACGAN
Implementation of *Auxiliary Classifier Generative Adversarial Network*.  

### BiGAN
Implementation of normal *Bidirectional Generative Adversarial Network*.  

