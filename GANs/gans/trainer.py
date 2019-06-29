from gans.config import *
from gans.generator import Generator
from gans.discriminator import Discriminator
from gans.gan import GAN
import gans.utils as utils

class Trainer:
    def __init__(self, output_size=(28,28,1), latent_size=100, epochs=5000, batch=32, checkpoint=50, model_type=-1):
        self.output_size = output_size
        self.latent_size = latent_size
        self.epochs = epochs
        self.batch = batch
        self.checkpoint = checkpoint
        self.model_type = model_type

        self.generator = Generator()
        self.discriminator = Discriminator()
        self.gan = GAN(generator_model = self.generator.model, discriminator_model=self.discriminator.model)

        self.load_data()

    def load_data(self):
        allowed_types = [-1,0,1,2,3,4,5,6,7,8,9]
        if self.model_type not in allowed_types:
            print('ERROR: Only Integer Values from -1 to 9 are allowed')
            raise Exception

        (self.X_train, self.Y_train), (_, _) = mnist.load_data()

        if self.model_type != -1:
            self.X_train = self.X_train[np.where(self.Y_train==int(self.model_type))[0]]

        # normalize
        self.X_train = ( np.float32(self.X_train) - 127.5) / 127.5
        self.X_train = np.expand_dims(self.X_train, axis=3)


    def gen_data_gen(self):
        while True:
            # fake
            no_fake = self.batch
            latent_tensor = self.sample_latent(no_fake)
            x = latent_tensor
            y = np.ones([no_fake, 1])

            yield x, y
    

    def disc_data_gen(self):
        for epoch in range(self.epochs):

            # MINI BATCH
            # real
            no_real = self.batch // 2
            i = randint(0, len(self.X_train)-no_real)
            real = self.X_train[ i : (i + no_real) ]
            y_real = np.ones([no_real,1])

            # fake
            no_fake = self.batch - no_real
            latent_tensor = self.sample_latent(no_fake)
            fake = self.generator.model(latent_tensor)
            y_fake = np.zeros([no_fake, 1])

            # concatenate real and fake examples
            x = np.concatenate([real, fake])
            y = np.concatenate([y_real, y_fake])

            yield x, y


    def train(self):
        # load data
        disc_data = self.disc_data_gen()
        gen_data = self.gen_data_gen()

        epoch = 0
        while True:
            try:
                # get discriminator data
                x_disc, y_disc = next(disc_data)

                # get generator data
                x_gen, y_gen = next(gen_data)

                # epoch
                epoch += 1
            except:
                break

            # train discriminator
            disc_loss = self.discriminator.model.train_on_batch(x_disc, y_disc)

            # train generator 
            gen_loss = self.gan.model.train_on_batch(x_gen, y_gen)

            print(f"Epoch {epoch}: \ndiscriminator loss: {disc_loss}\tgenerator loss: {gen_loss}")
            if epoch % self.checkpoint == 0:
                self.plot_checkpoint(epoch)


    def sample_latent(self, batch):
        return np.random.normal(0, 1,(batch,self.latent_size))


    def plot_checkpoint(self, epoch):
        checkpoint_dir = utils.mkdir(data_path/'checkpoints')
        filename = checkpoint_dir/(f"checkpoint{str(int(epoch))}.png")
        
        # generate images with generator from noise
        noise = self.sample_latent(9)
        images = self.generator.model.predict(noise)

        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            plt.subplot(3, 3, i+1)
            image = images[i, :, :, :]
            image = np.squeeze(image)
            plt.imshow(image, cmap='gray')
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(filename)
        plt.close('all')
