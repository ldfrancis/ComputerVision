
from gans.trainer import Trainer


output_size=(28,28,1)
latent_size=100
epochs=60000 
batch=32 
checkpoint=50 
model_type=-1


trainer = Trainer(output_size,
                 latent_size,
                 epochs,
                 batch,
                 checkpoint,
                 model_type)

trainer.train()