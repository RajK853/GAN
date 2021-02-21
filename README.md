# GAN

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Xh1dr_jE4rBBZv0HXTr7h3aO-a1cyO_p#scrollTo=NZlJZJKaQewN)  

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

### Train a model

```shell
python gan.py train ACGAN --epochs=100 --batch-size=128
```
The above command will train `ACGAN` model for 100 **epochs** with a **batch size** of 128.
> Use `--help` command to list out all the possible parameters.

### GAN
Implementation of normal *Generative Adversarial Network*.  

### ACGAN
Implementation of *Auxiliary Classifier Generative Adversarial Network*.  

### BiGAN
Implementation of normal *Bidirectional Generative Adversarial Network*.  

### Train models via YAML config file

1. Create a YAML config file (let's say `example_configs/config_1.yaml`) as:
```yaml  
default: &default_config
  epochs: 1000
  latent_size: 50
  batch_size: 128
  evaluate_interval: 5
  lr: 0.0003
  num_evaluates: 10

GAN_latent_50:
  <<: *default_config
  model: GAN

GAN_latent_100:
  <<: *default_config
  model: GAN
  latent_size: 100

ACGAN:
  <<: *default_config
  model: ACGAN

BiGAN:
  <<: *default_config
  model: BiGAN

```

2. Train the models by loading the parameters from the above YAML config file as:
```shell
python gan.py from-yaml example_configs/config_1.yaml
```
This will train each model using their parameters in the YAMl file. For instance, the above config file will train `GAN`, `ACGAN` and `BiGAN` models with two different `latent_size` values for the `GAN` model.  
>  Any configuration with the key name with the prefix `default` will not be executed by default.