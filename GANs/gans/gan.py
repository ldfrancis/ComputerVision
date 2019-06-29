from gans.config import *

class GAN(object):
    def __init__(self, discriminator_model, generator_model):
        # Initialize Variables
        self.generator = generator_model
        self.discriminator = discriminator_model
        self.discriminator.trainable = False
        self.model = self.build_model()
        self.model.compile(loss=binary_crossentropy, optimizer=Adam(lr=0.0002, decay=8e-9))
    
    def build_model(self):
        # Build the adversarial model and return it
        return Sequential([self.generator,
                            self.discriminator,])
    
    def summary(self):
        # Prints the Model Summary to the Screen
        return self.model.summary()
    
    def save_model(self):
        # Saves the model structure to a file in the data folder
        plot_model(self.model, to_file=models_path/'GAN_model.png')