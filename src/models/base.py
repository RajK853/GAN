import os
import numpy as np
from datetime import datetime
from collections import defaultdict

from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn


class BaseGAN:
    def __init__(self, dataset, img_shape, latent_size, log_prefix="log_", num_evaluates=10, console=None):
        self.dataset = dataset
        self.dataset_size = dataset.shape[0]
        self.img_shape = img_shape
        self.latent_size = latent_size
        self.num_evaluates = num_evaluates
        self.log_prefix = log_prefix
        self.history = defaultdict(list)
        
        current_time = datetime.now().strftime("%d.%m.%Y_%H.%M")
        self.log_dir = os.path.join("logs", f"{self.__class__.__name__}_{current_time}")
        self.img_log_path = os.path.join(self.log_dir, "images")
        os.makedirs(self.img_log_path, exist_ok=True)

        self.gen_images = []
        self.dis_model = None
        self.gen_model = None
        self.gan_mode = None

        self.data_index = 0
        self.indexes = np.arange(self.dataset_size, dtype="uint32")
        self.shuffle_indexes()

        self.console = console or Console()


    def load_generator(self, **kwargs):
        raise NotImplementedError

    def load_discriminator(self, **kwargs):
        raise NotImplementedError        
        
    def shuffle_indexes(self):
        np.random.shuffle(self.indexes)
        
    def sample_latent(self, batch_size):
        latent = np.random.randn(batch_size, self.latent_size)
        return latent
        
    def sample_real_data(self, batch_size):
        indexes = self.indexes[self.data_index:self.data_index+batch_size]
        data = self.dataset[indexes]
        self.data_index += batch_size
        if self.data_index >= self.dataset_size:
            self.data_index = 0
            self.shuffle_indexes()
        return data
    
    def sample_fake_data(self, batch_size):
        latent = self.sample_latent(batch_size)
        data = self.gen_model.predict(latent)
        return data
    
    def sample_data(self, batch_size):
        fake_data = self.sample_fake_data(batch_size)
        real_data = self.sample_real_data(batch_size)
        data = np.concatenate((fake_data, real_data), axis=0)
        labels = np.concatenate((np.zeros(fake_data.shape[0]), np.ones(real_data.shape[0])), axis=0)
        return data, labels

    def gather_logs(self, var_dict):
        def remove_prefix(key):
            return key[len(self.log_prefix):]
        return {remove_prefix(k):v for k, v in var_dict.items() if k.startswith(self.log_prefix)}

    def store_logs(self, log_dict):
        for k, v in log_dict.items():
            self.history[k].append(v)

    def train_once(self, batch_size):
        # Train discriminator
        batch_data, batch_labels = self.sample_data(batch_size)
        log_dis_loss, log_dis_acc = self.dis_model.train_on_batch(batch_data, batch_labels)
        # Train generator
        latent_vectors = self.sample_latent(2*batch_size)
        log_gen_loss = self.gan_model.train_on_batch(latent_vectors, np.ones((latent_vectors.shape[0], 1)))
        log_dict = self.gather_logs(vars())
        return log_dict

    def train(self, epochs=100, batch_size=32, evaluate_interval=10):
        # TODO: Is it really necessary to display all of these in the progress bar??
        train_per_epoch = np.math.ceil(self.dataset_size/batch_size)
        # p_bar = ProgressBar(total_iter=epochs*train_per_epoch, display_text="Training:", display_interval=10)
        step_text_fmt = (":: Epoch: [cyan]({task.fields[epoch]}/{task.fields[total_epoch]})[/cyan] :: dis (loss, acc): [cyan]({task.fields[dis_loss]:.4f}, {task.fields[dis_acc]:.4f})[/cyan] "
                         ":: gen loss: [cyan]{task.fields[gen_loss]:.4f}[/cyan]")
        p_bar = Progress(TextColumn("{task.description}"),
                         BarColumn(complete_style="bold yellow", finished_style="bold cyan"),
                         "[progress.percentage]{task.percentage:>3.1f}%",
                         ":: Time left:",
                         TimeRemainingColumn(),
                         TextColumn(step_text_fmt),
                         refresh_per_second=2)
        print()
        self.console.rule(f"Training for [cyan]{epochs}[/cyan] epochs with [cyan]{train_per_epoch}[/cyan] iterations per epoch")
        with p_bar:
            task = p_bar.add_task(f" Training:", total=epochs*train_per_epoch, visible=False, epoch=0, total_epoch=epochs, dis_loss=0, dis_acc=0, gen_loss=0)
            for epoch in range(1, epochs+1):
                for batch_iter in range(train_per_epoch):
                    log_dict = self.train_once(batch_size)
                    self.store_logs(log_dict)
                    p_bar.update(task, advance=1, visible=True, epoch=epoch, **log_dict)
                    # p_bar.step(add_text=step_text_fmt.format(epoch=epoch, **log_dict))
                if (epoch == epochs) or (epoch%evaluate_interval == 0):
                    self.save_images(epoch)
        meta_data = {"epochs": epochs, "latent_size": self.latent_size, "batch_size": batch_size, "evaluate_interval": evaluate_interval}
        print()
        self.console.rule("Saving logs and models")
        self.save_meta_data(meta_data)
        self.save_models()
        self.save_log()
        self.console.rule("Training finished")

    def save_images(self, epoch, cols=5):
        import matplotlib.pyplot as plt
        gen_imgs = self.sample_fake_data(self.num_evaluates)
        rows = np.math.ceil(self.num_evaluates/cols)
        fig = plt.figure(figsize=(20, 4*rows))       # Each row has the size of 4
        for i, img in enumerate(gen_imgs, start=1):
            fig.add_subplot(rows, cols, i)
            plt.imshow(img[:, :, 0], cmap="gray")
        fig.savefig(os.path.join(self.img_log_path, f"Epoch_{epoch}"))
        plt.close()

    def save_models(self):
        dump_path = os.path.join(self.log_dir, "models")
        os.makedirs(dump_path, exist_ok=True)
        self.gen_model.save(os.path.join(dump_path, "generator.h5"))
        self.dis_model.save(os.path.join(dump_path, "discriminator.h5"))
        self.gan_model.save(os.path.join(dump_path, "combined.h5"))
        self.console.print(f"# Models saved in [cyan]'{dump_path}'[/cyan] directory.")

    def save_log(self):
        from pandas import DataFrame
        df = DataFrame.from_dict(self.history, dtype=np.float32)
        dump_path = os.path.join(self.log_dir, "history.csv")
        df.to_csv(dump_path, index=False)
        self.console.print(f"# Training history saved to [cyan]'{dump_path}'[/cyan].")

    def save_meta_data(self, data_dict):
        import json
        dump_path = os.path.join(self.log_dir, "meta_data.json")
        with open(dump_path, "w") as outfile:
            json.dump(data_dict, outfile, indent=True)
        self.console.print(f"# Meta data dumped to [cyan]'{dump_path}'[/cyan].")
