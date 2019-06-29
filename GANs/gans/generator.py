from gans.config import *

class Generator(object):

    start_size = 128
    
    def __init__(self, output_size=(28,28,1), latent_size=100):
        # Initialize Variables
        self.output_size = output_size
        self.latent_size = latent_size
        self.model = self.build_model()
        self.model.compile(loss=binary_crossentropy, optimizer=Adam(lr=0.0002, decay=8e-9))
        


    def block(self,x):

        x = Dense(self.start_size)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(momentum=0.8)(x)

        return x

    def build_model(self, start_size=128, num_blocks=4):
        # Build the generator model and returns it

        self.start_size = start_size

        inp = Input(shape=(self.latent_size,))
        x = self.block(inp)

        for i in range(num_blocks):
            x = self.block(x)
        x = Dense(np.prod(self.output_size), activation='tanh')(x)
        y = Reshape(self.output_size)(x)

        return Model(inputs=[inp], outputs=[y])
    
    def summary(self):
        # Prints the Model Summary to the screen

        return self.model.summary()
    
    def save_model(self):
        # Saves the model structure to a file in the data folder
        plot_model(self.model,to_file=models_path/'gen_model.png')