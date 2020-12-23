from tensorflow.compat.v1.keras import layers, optimizers, Model, Input

from . import BaseGAN


class GAN(BaseGAN):
    def __init__(self, *args, **kwargs):
        super(GAN, self).__init__(*args, **kwargs)
        self.latent_in = Input(shape=self.latent_size, name="latent_input", dtype="float32")
        self.img_in = Input(shape=self.img_shape, name="image_input", dtype="float32")
        
        self.dis_model = self.load_discriminator()
        opt = optimizers.Adam(learning_rate=3e-4, beta_1=0.5)
        self.dis_model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["acc"])
        
        self.gen_model = self.load_generator()
        
        self.dis_model.trainable = False
        img_out = self.gen_model([self.latent_in])
        prob_out = self.dis_model(img_out)
        self.gan_model = Model(inputs=[self.latent_in], outputs=[prob_out])
        self.gan_model.compile(optimizer=opt, loss="binary_crossentropy")
        
    def load_generator(self):
        x = layers.Dense(7*7*64, activation="relu", kernel_regularizer="l2")(self.latent_in)
        x = layers.Reshape((7, 7, 64))(x)
        x = layers.Conv2D(filters=64, kernel_size=3, activation="relu", padding="same", kernel_regularizer="l2")(x)
        x = layers.UpSampling2D()(x)
        x = layers.Conv2DTranspose(filters=32, kernel_size=5, activation="relu", padding="same", kernel_regularizer="l2")(x)
        x = layers.UpSampling2D()(x)
        img = layers.Conv2D(filters=1, kernel_size=3, padding="same", activation="sigmoid")(x)
        model = Model(inputs=[self.latent_in], outputs=[img])
        return model
    
    def load_discriminator(self, opt="adam"):
        x = layers.Conv2D(filters=32, kernel_size=3, activation="relu", kernel_regularizer="l2")(self.img_in)
        x = layers.Conv2D(filters=64, kernel_size=5, activation="relu", kernel_regularizer="l2")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(100, activation="relu", kernel_regularizer="l2")(x)
        x = layers.Dense(100, activation="relu", kernel_regularizer="l2")(x)
        """x = layers.Dense(32, activation="relu", kernel_regularizer="l2")(x)
        sx = layers.Dense(32, activation="tanh", kernel_regularizer="l2")(x)
        x = layers.Multiply()([sx, x])"""
        probs = layers.Dense(1, activation="sigmoid")(x)
        model = Model(inputs=[self.img_in], outputs=[probs])
        return model
    