import numpy as np
from tensorflow.compat.v1.keras import layers, optimizers, Model, Input

from . import BaseGAN


class BiGAN(BaseGAN):
    def __init__(self, *args, **kwargs):
        super(BiGAN, self).__init__(*args, **kwargs)
        self.progress_fmt = "Epoch: ({epoch}/{total_epoch}) | dis (loss, acc): ({dis_loss:.4f}, {dis_acc:.4f}) | gen loss: {gen_loss:.4f}"
        # Initialize inputs
        self.label_in = Input(shape=(1, ), name="label_input", dtype="int32")
        self.latent_in = Input(shape=(self.latent_size, ), name="latent_input", dtype="float32")
        self.img_in = Input(shape=self.img_shape, name="image_input", dtype="float32")
        # Initialize optimizers
        dis_opt = optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.5, name="discriminator_opt")
        combined_opt = optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.5, name="combined_opt")
        # Load models
        self.encoder_model = self.load_encoder(self.img_in)
        self.gen_model = self.load_generator(self.latent_in)
        self.dis_model = self.load_discriminator(self.img_in, self.latent_in, opt=dis_opt)
        self.combined_model = self.load_combined_model(self.img_in, self.latent_in, opt=combined_opt)
        self.models = [self.gen_model, self.dis_model, self.combined_model, self.encoder_model]

    def load_encoder(self, img_in):
        x = layers.Conv2D(filters=16, kernel_size=3, kernel_regularizer="l2")(img_in)
        x = layers.MaxPool2D()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters=8, kernel_size=3, activation="relu", kernel_regularizer="l2")(x)
        x = layers.Flatten()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(64, activation="relu", kernel_regularizer="l2")(x)
        latent_out = layers.Dense(self.latent_size, name="latent_output")(x)
        model = Model(inputs=[img_in], outputs=[latent_out], name="encoder")
        return model

    def load_generator(self, latent_in):
        x = layers.Dense(64, kernel_regularizer="l2")(latent_in)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dense(7*7*4, activation="relu", kernel_regularizer="l2")(x)
        x = layers.Reshape((7, 7, 4))(x)
        x = layers.Conv2D(filters=4, kernel_size=3, activation="relu", padding="same", kernel_regularizer="l2")(x)
        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(filters=4, kernel_size=3, activation="relu", padding="same", kernel_regularizer="l2")(x)
        x = layers.UpSampling2D()(x)
        img_out = layers.Conv2D(filters=1, kernel_size=3, padding="same", activation="sigmoid", name="image_output")(x)
        model = Model(inputs=[latent_in], outputs=[img_out], name="generator")
        return model
    
    def load_discriminator(self, img_in, latent_in, opt="adam"):
        x = layers.Conv2D(filters=8, kernel_size=3, kernel_regularizer="l2")(img_in)
        x = layers.MaxPool2D()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters=16, kernel_size=3, activation="relu", kernel_regularizer="l2")(x)
        x = layers.Flatten()(x)
        x = layers.Concatenate()([latent_in, x])
        x = layers.Dense(64, activation="relu", kernel_regularizer="l2")(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(32, activation="relu", kernel_regularizer="l2")(x)
        label_out = layers.Dense(1, activation="sigmoid", name="label_output")(x)
        model = Model(inputs=[img_in, latent_in], outputs=[label_out], name="discriminator")
        model.compile(optimizer=opt, loss=["binary_crossentropy"], metrics=["acc"])
        return model

    def load_combined_model(self, img_in, latent_in, opt="adam"):
        self.dis_model.trainable = False
        # P(E(x)|x,z)
        latent_out = self.encoder_model(img_in)
        encod_label_out = self.dis_model([img_in, latent_out])
        # P(G(z)|x,z)
        img_out = self.gen_model(latent_in)
        gen_label_out = self.dis_model([img_out, latent_in])
        
        model = Model(inputs=[img_in, latent_in], outputs=[encod_label_out, gen_label_out], name="combined")
        model.compile(optimizer=opt, loss=["binary_crossentropy", "binary_crossentropy"])
        return model

    def generate_images(self, batch_size):
        latent_vectors = self.sample_latent(batch_size)
        gen_imgs = self.gen_model.predict([latent_vectors])
        return gen_imgs

    def generate_feed_dict(self, batch_size):
        # Generate required data
        indexes = self.indexes[self.data_index:self.data_index+batch_size]
        batch_size = min(len(indexes), batch_size)

        real_imgs = self.dataset[indexes]

        latent_vectors = self.sample_latent(batch_size)
        fake_imgs = self.gen_model.predict([latent_vectors])

        encoded_latent_vectors = self.encoder_model.predict(real_imgs)
        ones_labels = np.ones((batch_size, 1))
        zeros_labels = np.zeros((batch_size, 1))
        # Clear batch_feed_dict
        self._batch_feed_dict.clear()
        # Update feed_data_dict with discriminator inputs and outputs
        self._batch_feed_dict["dis_inputs"] = [np.concatenate([fake_imgs, real_imgs], axis=0), np.concatenate([latent_vectors, encoded_latent_vectors], axis=0)]
        self._batch_feed_dict["dis_outputs"] = [np.concatenate([zeros_labels, ones_labels], axis=0)]
        # TODO: Does it makes sense to sample additional real image data to train the encoder_model on 2*batch_size?
        # (Combined) generator inputs and outputs
        self._batch_feed_dict["gen_inputs"] = [real_imgs, latent_vectors]
        self._batch_feed_dict["gen_outputs"] = [zeros_labels, ones_labels]   # Combined_layer outputs = [P(E(z)=1|x, z), P(G(z)=1|x, z)]?? TODO: Confirm it

    def process_dis_result(self, result):
        dis_loss, dis_acc = result
        return {"dis_loss": dis_loss, "dis_acc": dis_acc}

    def process_gen_result(self, result):
        gen_loss, *_ = result
        return {"gen_loss": gen_loss}
