import sys
from colorama import init, Fore, Back, Style

from src import models as gan_models                  ## Use importlib instead?
from src.utils import load_mnist_data, exec_from_yaml

init(autoreset=True)


def main(model, epochs=100, batch_size=128, evaluate_interval=10, **kwargs):
    print(f"{Fore.YELLOW}Loading MNIST data")
    (mnist_data, mnist_labels), *_ = load_mnist_data()
    print()
    if model not in gan_models.__all__:
        print(f"{Fore.YELLOW}Model {Fore.RED}{model}{Style.RESET_ALL} not found. Valid models {Fore.RED}{gan_models.__all__}")
    else:
        ModelClass = getattr(gan_models, model)
        gan = ModelClass(mnist_data, mnist_labels, img_shape=mnist_data[0].shape, **kwargs)
        gan.train(epochs=epochs, batch_size=batch_size, evaluate_interval=evaluate_interval)


if __name__ == "__main__":
    config_path = sys.argv[1]
    exec_from_yaml(config_path, exec_func=main, title="Configuration", safe_load=True, ignore_exp_prefix="default")