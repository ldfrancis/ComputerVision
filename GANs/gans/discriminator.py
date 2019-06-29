from gans.config import *

class Discriminator(object):
    def __init__(self, input_size=(28,28), channels = 1, latent_size=100):
        # Initialize Variables
        
        self.input_shape = input_size+(channels,)
        self.optimizer = Adam(lr=0.0002, decay=8e-9)
        self.capacity = np.prod(self.input_shape)
        self.model = self.build_model()
        self.model.compile(loss = binary_crossentropy, optimizer=self.optimizer)
        
    
    def build_model(self):
        # Build the binary classifier and return it

        inp = Input(shape=self.input_shape)

        x = Flatten(name="flatten")(inp)
        x = Dense(self.capacity, name="dense1")(x)
        x = LeakyReLU(alpha=0.2, name="leaky_relu1")(x)
        x = Dense(self.capacity//2, name="dense2")(x)
        x = LeakyReLU(alpha=0.2, name="leaky_relu2")(x)

        y = Dense(1, activation='sigmoid')(x)

        return Model(inputs=[inp], outputs=[y])
    
    def summary(self):
        # Prints the Model Summary to the Screen

        return self.model.summary()

    
    def save_model(self):
        # Saves the model structure to a file in the data folder
        plot_model(self.model, to_file=models_path/'disc_model.png')
