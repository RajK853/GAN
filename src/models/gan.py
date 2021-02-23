import numpy as np
from tensorflow.compat.v1.keras import layers, optimizers, Model, Input

from . import BaseGAN
from .base import DEFAULT_CONFIGS
from ..utils import create_feedforward_network

class GAN(BaseGAN):
    def __init__(self, *args, **kwargs):
        super(GAN, self).__init__(*args, **kwargs)
        self.progress_fmt = "Epoch: ({epoch}/{total_epoch}) | dis (loss, acc): ({dis_loss:.4f}, {dis_acc:.4f}) | gen loss: {gen_loss:.4f}"
        # Initialize inputs
        self.latent_in = Input(shape=(self.latent_size, ), name="latent_input", dtype="float32")
        self.img_in = Input(shape=self.img_shape, name="image_input", dtype="float32")
        # Initialize optimizers
        dis_opt = optimizers.Adam(learning_rate=self.lr, beta_1=0.5, name="discriminator_opt")
        combined_opt = optimizers.Adam(learning_rate=self.lr, beta_1=0.5, name="combined_opt")
        # Load models
        self.gen_model = self.load_generator(self.latent_in)
        self.dis_model = self.load_discriminator(self.img_in, dis_opt)
        self.combined_model = self.load_combined_model(self.latent_in, combined_opt)
        self.models = [self.gen_model, self.dis_model, self.combined_model]

    def load_generator(self, latent_in):
        layer_config = self.layer_configs.get("generator", None)
        if layer_config is None:
            layer_config = DEFAULT_CONFIGS["generator"]
            print("Loading default generator network!")
        x = create_feedforward_network(latent_in, layers=layer_config)
        img_out = layers.Conv2D(filters=1, kernel_size=5, padding="same", activation="sigmoid", name="image_output")(x)
        model = Model(inputs=[latent_in], outputs=[img_out], name="generator")
        return model
    
    def load_discriminator(self, img_in, opt="adam"):
        layer_config = self.layer_configs.get("discriminator", None)
        if layer_config is None:
            layer_config = DEFAULT_CONFIGS["discriminator"]
            print("Loading default discriminator network!")
        x = create_feedforward_network(img_in, layers=layer_config)
        label_out = layers.Dense(1, activation="sigmoid", name="binary_prob_output")(x)
        model = Model(inputs=[img_in], outputs=[label_out], name="discriminator")
        model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["acc"])
        return model

    def load_combined_model(self, latent_in, opt="adam"):
        self.dis_model.trainable = False
        img_out = self.gen_model(latent_in)
        label_out = self.dis_model(img_out)
        model = Model(inputs=[latent_in], outputs=[label_out], name="combined")
        model.compile(optimizer=opt, loss="binary_crossentropy")
        return model

    def generate_images(self, batch_size):
        latent_vectors = self.sample_latent(batch_size)
        gen_imgs = self.gen_model.predict(latent_vectors)
        return gen_imgs

    def generate_feed_dict(self, batch_size):
        # Generate required data
        indexes = self.indexes[self.data_index:self.data_index+batch_size]
        batch_size = min(len(indexes), batch_size)
        real_imgs = self.dataset[indexes]
        latent_vectors = self.sample_latent(batch_size)
        fake_imgs = self.gen_model.predict(latent_vectors)
        ones_labels = np.ones((batch_size, 1))
        zeros_labels = np.zeros((batch_size, 1))
        # Clear batch_feed_dict
        self._batch_feed_dict.clear()
        # Update feed_data_dict with discriminator inputs and outputs
        self._batch_feed_dict["dis_inputs"] = [np.concatenate([fake_imgs, real_imgs], axis=0)]
        self._batch_feed_dict["dis_outputs"] = [np.concatenate([zeros_labels, ones_labels], axis=0)]
        # As the discriminator is trained on generated and sampled batches with each's size = batch_size,
        # we are generating extra batches here to also train the generator on total of 2*batch_size.
        extra_latent_vectors = self.sample_latent(batch_size)
        gan_latent_vectors = np.concatenate([latent_vectors, extra_latent_vectors], axis=0)
        # Update feed_data_dict with combined_model inputs and outputs
        self._batch_feed_dict["gen_inputs"] = [gan_latent_vectors]
        self._batch_feed_dict["gen_outputs"] = [np.ones((2*batch_size, 1))]
    
    def process_dis_result(self, result):
        dis_loss, dis_acc = result
        return {"dis_loss": dis_loss, "dis_acc": dis_acc}

    def process_gen_result(self, result):
        return {"gen_loss": result}
