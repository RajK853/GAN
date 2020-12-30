import os
import numpy as np
from time import time
from datetime import datetime
from collections import defaultdict
from colorama import init, Fore, Back, Style
from tensorflow.compat.v1.keras import callbacks, backend

from src.progressbar import ProgressBar


class BaseGAN:
    def __init__(self, dataset, class_labels, img_shape, latent_size, num_evaluates=10):
        self.dataset = dataset
        self.class_labels = class_labels
        self.dataset_size = dataset.shape[0]
        self.img_shape = img_shape
        self.latent_size = latent_size
        self.num_evaluates = num_evaluates
        self.history = defaultdict(list)
        # Init log directory
        current_time = datetime.now().strftime("%d.%m.%Y_%H.%M")
        self.log_dir = os.path.join("logs", f"{self.__class__.__name__}_{current_time}")
        self.img_log_path = os.path.join(self.log_dir, "images")
        self.tensorboard_log_path = os.path.join(self.log_dir, "tensorboard")
        self.tensorboard_callback = callbacks.TensorBoard(self.tensorboard_log_path)
        os.makedirs(self.img_log_path, exist_ok=True)
        # Models object references
        self.models = []
        self.dis_model = None
        self.gen_model = None
        self.combined_mode = None
        # Data sampler variables
        self.data_index = 0
        self.indexes = np.arange(self.dataset_size, dtype="uint32")
        self.shuffle_indexes()
        self._batch_feed_dict = {}

    def load_generator(self, **kwargs):
        raise NotImplementedError

    def load_discriminator(self, **kwargs):
        raise NotImplementedError

    def load_combined_model(self, **kwargs):
        raise NotImplementedError

    def process_dis_result(self, result):
        """
        Process results obtained from training discriminator and return as a dictionary
        """
        raise NotImplementedError

    def process_gen_result(self, result):
        """
        Process results obtained from training combined_network and return as a dictionary
        """
        raise NotImplementedError

    def generate_images(self, batch_size):
        raise NotImplementedError

    def generate_feed_dict(self, batch_size):
        raise NotImplementedError
        
    def shuffle_indexes(self):
        np.random.shuffle(self.indexes)
        
    def sample_latent(self, batch_size):
        latent = np.random.randn(batch_size, self.latent_size)
        return latent

    def increment_index(self, batch_size):
        self.data_index += batch_size
        if self.data_index >= self.dataset_size:
            self.data_index = 0

    def _train_discriminator(self):
        result = self.dis_model.train_on_batch(self._batch_feed_dict["dis_inputs"], self._batch_feed_dict["dis_outputs"])
        result_dict = self.process_dis_result(result)
        return result_dict

    def _train_generator(self):
        result = self.combined_model.train_on_batch(self._batch_feed_dict["gen_inputs"], self._batch_feed_dict["gen_outputs"])
        result_dict = self.process_gen_result(result)
        return result_dict

    def train_once(self, batch_size):
        self.generate_feed_dict(batch_size)
        dis_result = self._train_discriminator()
        gen_result = self._train_generator()
        self.increment_index(batch_size)
        return {**dis_result, **gen_result}

    def train(self, epochs=100, batch_size=32, evaluate_interval=10):
        start_time = time()
        for model in self.models:
            self.tensorboard_callback.set_model(model)
        train_per_epoch = np.math.ceil(self.dataset_size/batch_size)
        batch_iter_progress = 1/train_per_epoch
        p_bar = ProgressBar(total_iter=epochs*train_per_epoch, title=f"{Fore.CYAN}Training:{Style.RESET_ALL}", display_interval=10, info_text=self.progress_fmt)
        print(f"Training {Fore.YELLOW}{self.__class__.__name__} {Style.RESET_ALL} for {Fore.CYAN}{epochs}{Style.RESET_ALL} epochs with {Fore.CYAN}{train_per_epoch}{Style.RESET_ALL} iterations per epoch")        
        for epoch in range(1, epochs+1):
            self.shuffle_indexes()
            for batch_iter in range(train_per_epoch):
                log_dict = self.train_once(batch_size)
                # TODO: Implement code to log train results
                self.tensorboard_callback.on_train_batch_end(batch_iter, logs=log_dict)
                self.store_logs(log_dict)
                p_bar.step(epoch=epoch, total_epoch=epochs, **log_dict)
            if (epoch == epochs) or (epoch%evaluate_interval == 0):
                self.save_images(epoch)
        print("\nSaving logs and models")
        meta_data = {"epochs": epochs, "latent_size": self.latent_size, "batch_size": batch_size, "evaluate_interval": evaluate_interval}
        self.save_meta_data(meta_data)
        self.save_models()
        self.save_log()
        backend.clear_session()
        elapsed_time = (time() - start_time)/60
        print(f"\nTraining finished in {Fore.CYAN}{elapsed_time:.1f}{Style.RESET_ALL} minutes\n")

    def store_logs(self, log_dict):
        for k, v in log_dict.items():
            self.history[k].append(v)

    def save_images(self, epoch, cols=5):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        gen_imgs = self.generate_images(self.num_evaluates)
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
        for model in self.models:
            model.save(os.path.join(dump_path, f"{model.name}.h5"))
        print(f"# Models saved in {Fore.CYAN}'{dump_path}'{Style.RESET_ALL} directory")

    def save_log(self):
        from pandas import DataFrame
        df = DataFrame.from_dict(self.history, dtype=np.float32)
        dump_path = os.path.join(self.log_dir, "history.csv")
        df.to_csv(dump_path, index=False)
        print(f"# Training history saved to {Fore.CYAN}'{dump_path}'")

    def save_meta_data(self, data_dict):
        import json
        # TODO: Save network structures too i.e. input and output info?
        dump_path = os.path.join(self.log_dir, "meta_data.json")
        with open(dump_path, "w") as outfile:
            json.dump(data_dict, outfile, indent=True)
        print(f"# Meta data dumped to {Fore.CYAN}'{dump_path}'")
