from GAN import *
from dataset import parse_dataset

def trainer():
    x_train = parse_dataset()[0][0]

    batch_size = 64
    epochs = 100
    d_lr = 0.0002
    g_lr = 0.0002
    beta_1 = 0.5
    d_thresh = 0.8
    soft_label = False
    adv_weight = 1

    # Build and compile discriminator and generator models

    d_loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
    g_loss_fn = keras.losses.MeanAbsoluteError()

    d_optimizer = keras.optimizers.Adam(learning_rate=d_lr, beta_1=beta_1)
    g_optimizer = keras.optimizers.Adam(learning_rate=g_lr, beta_1=beta_1)

    # Define accuracy metric for discriminator
    def discriminator_accuracy(y_true, y_pred):
        return keras.metrics.binary_accuracy(tf.round(y_true), tf.round(y_pred))

    # Define training loop
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch+1, epochs))
        
        for phase in ['train']:
            if phase == 'train':
                D.trainable = True
                G.trainable = True
            else:
                D.trainable = False
                G.trainable = False
            
            running_loss_G = 0.0
            running_loss_D = 0.0
            running_loss_adv_G = 0.0
            running_acc_D = 0.0
            
            for i, X in enumerate(tqdm(dset_loaders[phase])):
                
                X = X.astype('float32')
                batch = X.shape[0]
                
                # Train discriminator
                with tf.GradientTape() as tape:
                    d_real = D(X)
                    
                    Z = np.random.normal(size=(batch, 100))
                    fake = G(Z)
                    d_fake = D(fake)
                    
                    real_labels = tf.ones_like(d_real)
                    fake_labels = tf.zeros_like(d_fake)
                    
                    if soft_label:
                        real_labels = tf.random.uniform(shape=(batch,), minval=0.7, maxval=1.2)
                        fake_labels = tf.random.uniform(shape=(batch,), minval=0.0, maxval=0.3)
                    
                    d_real_loss = d_loss_fn(real_labels, d_real)
                    d_fake_loss = d_loss_fn(fake_labels, d_fake)
                    d_loss = d_real_loss + d_fake_loss
                    
                    d_real_acc = discriminator_accuracy(real_labels, d_real)
                    d_fake_acc = discriminator_accuracy(fake_labels, d_fake)
                    d_acc = 0.5 * (d_real_acc + d_fake_acc)
                    
                    if d_acc < d_thresh:
                        grads = tape.gradient(d_loss, D.trainable_weights)
                        d_optimizer.apply_gradients(zip(grads, D.trainable_weights))
                    
                    running_loss_D += d_loss.numpy() * batch
                    running_acc_D += d_acc.numpy() * batch
                
                # Train generator
                with tf.GradientTape() as tape:
                    Z = np.random.normal(size=(batch, 100))
                    fake = G(Z)
                    d_fake = D(fake)
                    
                    adv_g_loss = d_loss_fn(tf.ones_like(d_fake), d_fake)
                    recon_g_loss = g_loss_fn(X, fake)
                    g_loss = recon_g_loss + adv_weight * adv_g_loss
                    
                    grads = tape.gradient(g_loss, G.trainable_weights)
                    g_optimizer.apply_gradients(zip(grads, G.trainable_weights))
                    
                    running_loss_G += recon_g_loss.numpy() * batch
                    running_loss_adv_G += adv_g_loss.numpy() * batch