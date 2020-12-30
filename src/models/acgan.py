import numpy as np
from tensorflow.compat.v1.keras import layers, optimizers, Model, Input

from . import BaseGAN


class ACGAN(BaseGAN):
    def __init__(self, *args, **kwargs):
        super(ACGAN, self).__init__(*args, **kwargs)
        self.progress_fmt = ("Epoch: ({epoch}/{total_epoch}) | dis (loss, acc, label_acc): ({dis_loss:.3f}, {dis_acc:.3f}, {dis_label_acc:.3f}) | gen loss: {gen_loss:.4f}")
        self.num_classes = len(set(self.class_labels.squeeze()))
        # Initialize inputs
        self.label_in = Input(shape=(1, ), name="label_input", dtype="int32")
        self.latent_in = Input(shape=(self.latent_size, ), name="latent_input", dtype="float32")
        self.img_in = Input(shape=self.img_shape, name="image_input", dtype="float32")
        # Initialize optimizers
        dis_opt = optimizers.Adam(learning_rate=3e-4, beta_1=0.5, name="discriminator_opt")
        combined_opt = optimizers.Adam(learning_rate=3e-4, beta_1=0.5, name="combined_opt")
        # Load models
        self.gen_model = self.load_generator(self.latent_in, self.label_in)
        self.dis_model = self.load_discriminator(self.img_in, opt=dis_opt)
        self.combined_model = self.load_combined_model(self.latent_in, self.label_in, opt=combined_opt)
        self.models = [self.gen_model, self.dis_model, self.combined_model]

    def load_generator(self, latent_in, label_in):
        # Embedd given label into given matrix
        ex = layers.Embedding(self.num_classes, self.latent_size)(label_in)
        ex = layers.Flatten()(ex)
        x = layers.Multiply()([latent_in, ex])                             # Combine embedded label and latent vectors
        x = layers.Dense(7*7*32, activation="relu", kernel_regularizer="l2")(x)
        x = layers.Reshape((7, 7, 32))(x)
        x = layers.Conv2D(filters=32, kernel_size=5, activation="relu", padding="same", kernel_regularizer="l2")(x)
        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(filters=16, kernel_size=3, activation="relu", padding="same", kernel_regularizer="l2")(x)
        x = layers.UpSampling2D()(x)
        img = layers.Conv2D(filters=1, kernel_size=3, padding="same", activation="sigmoid", name="gen_image_output")(x)
        model = Model(inputs=[latent_in, label_in], outputs=[img], name="generator")
        return model
    
    def load_discriminator(self, img_in, opt="adam"):
        x = layers.Conv2D(filters=32, kernel_size=3, activation="relu", kernel_regularizer="l2")(img_in)
        x = layers.Conv2D(filters=16, kernel_size=5, activation="relu", kernel_regularizer="l2")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation="relu", kernel_regularizer="l2")(x)
        x = layers.Dense(32, activation="relu", kernel_regularizer="l2")(x)
        label_out = layers.Dense(1, activation="sigmoid", name="label_output")(x)
        class_out = layers.Dense(self.num_classes, activation="softmax", name="class_output")(x)
        model = Model(inputs=[img_in], outputs=[label_out, class_out], name="discriminator")
        model.compile(optimizer=opt, loss=["binary_crossentropy", "sparse_categorical_crossentropy"], metrics=["acc"])
        return model

    def load_combined_model(self, latent_in, label_in, opt="adam"):
        self.dis_model.trainable = False
        img_out = self.gen_model([latent_in, label_in])
        label_out, class_out = self.dis_model(img_out)
        model = Model(inputs=[latent_in, label_in], outputs=[label_out, class_out], name="combined")
        model.compile(optimizer=opt, loss=["binary_crossentropy", "sparse_categorical_crossentropy"])
        return model

    def sample_random_class_labels(self, batch_size):
        labels = np.random.randint(0, self.num_classes, size=(batch_size, 1))
        return labels

    def generate_images(self, batch_size):
        latent_vectors = self.sample_latent(batch_size)
        labels = self.sample_random_class_labels(batch_size)
        gen_imgs = self.gen_model.predict([latent_vectors, labels])
        return gen_imgs

    def generate_feed_dict(self, batch_size):
        # Generate required data
        indexes = self.indexes[self.data_index:self.data_index+batch_size]
        batch_size = min(len(indexes), batch_size)

        real_imgs = self.dataset[indexes]
        real_class_labels = self.class_labels[indexes]

        latent_vectors = self.sample_latent(batch_size)
        fake_class_labels = self.sample_random_class_labels(batch_size)
        fake_imgs = self.gen_model.predict([latent_vectors, fake_class_labels])

        ones_labels = np.ones((batch_size, 1))
        zeros_labels = np.zeros((batch_size, 1))
        # Clear batch_feed_dict
        self._batch_feed_dict.clear()
        # Update feed_data_dict with discriminator inputs and outputs
        self._batch_feed_dict["dis_inputs"] = [np.concatenate([fake_imgs, real_imgs], axis=0)]
        self._batch_feed_dict["dis_outputs"] = [np.concatenate([zeros_labels, ones_labels], axis=0), np.concatenate([fake_class_labels, real_class_labels])]
        # As the discriminator is trained on generated and sampled batches with each's size = batch_size,
        # we are generating extra batches here to also train the generator on total of 2*batch_size.
        extra_latent_vectors = self.sample_latent(batch_size)
        extra_fake_class_labels = self.sample_random_class_labels(batch_size)
        gen_latent_vectors = np.concatenate([latent_vectors, extra_latent_vectors], axis=0)
        gen_class_labels = np.concatenate([fake_class_labels, extra_fake_class_labels], axis=0)
        # Update feed_data_dict with combined_model inputs and outputs
        self._batch_feed_dict["gen_inputs"] = [gen_latent_vectors, gen_class_labels]
        self._batch_feed_dict["gen_outputs"] = [np.ones((2*batch_size, 1)), gen_class_labels]

    def process_dis_result(self, result):
        dis_loss, *_, dis_acc, dis_label_acc = result
        return {"dis_loss": dis_loss, "dis_acc": dis_acc, "dis_label_acc": dis_label_acc}

    def process_gen_result(self, result):
        gen_loss, *_ = result
        return {"gen_loss": gen_loss}
