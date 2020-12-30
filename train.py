import typer
from typing import List
from src import models as gan_models   ## Use importlib instead?
from colorama import init, Fore, Back, Style

init(autoreset=True)
app = typer.Typer()


@app.command()
def main(models: List[str], epochs: int = 100, latent_size: int = 50, batch_size: int = 128, evaluate_interval: int = 10, num_evaluates: int = 10):
    from src.utils.dataset import load_mnist_data
    print(f"{Fore.YELLOW}Loading MNIST data")
    (mnist_data, mnist_labels), *_ = load_mnist_data()
    img_shape = mnist_data[0].shape
    print()
    for model in models:
        if model not in gan_models.__all__:
            print(f"{Fore.YELLOW}Model {Fore.RED}{model}{Style.RESET_ALL} not found. Valid models {Fore.RED}{gan_models.__all__}")
            continue
        ModelClass = getattr(gan_models, model)
        gan = ModelClass(mnist_data, mnist_labels, img_shape=img_shape, latent_size=latent_size, num_evaluates=num_evaluates)
        gan.train(epochs=epochs, batch_size=batch_size, evaluate_interval=evaluate_interval)


if __name__ == "__main__":
    app()