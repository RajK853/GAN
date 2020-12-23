import typer
from rich.console import Console
from src import models as gan_models

console = Console()
app = typer.Typer()


@app.command()
def run(model: str = "gan", epochs: int = 100, latent_size: int = 50, batch_size: int = 128, evaluate_interval: int = 10):
    from src.utils.dataset import load_mnist_data
    console.rule(f"[bold yellow]{model.upper()}")
    if model not in gan_models.__all__:
        console.print(f"[yellow]Model [dim red]{model}[/dim red] not found. Valid models [dim red]{gan_models.__all__}[/dim red].")
        raise typer.Exit()

    print()
    console.rule("Loading MNIST data")
    (mnist_data, _), *_ = load_mnist_data()
    img_shape = mnist_data[0].shape
    ModelClass = getattr(gan_models, model)
    gan = ModelClass(mnist_data, img_shape, latent_size)
    gan.train(epochs=epochs, batch_size=batch_size, evaluate_interval=evaluate_interval)


if __name__ == "__main__":
    app()