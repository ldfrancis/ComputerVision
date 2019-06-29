import tensorflow as tf

def update(model,x,y,loss,optim):
        
    with tf.GradientTape() as tape:
        logits = model(x)
        loss_value = loss(logits, y)
    grads = tape.gradient(loss_value, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))

def mkdir(name):
    
    if not name.exists():
        name.mkdir()

    return name