import tensorflow as tf
from model import WarpingGAN, Discriminator

from gradient_penalty import GradientPenalty
from data_benchmark import BenchmarkDataset
from stitchingloss import stitchloss

from arguments import Arguments

import time

class WarpingGANTrain():
    def __init__(self, args):
        self.args = args
        
        self.data = BenchmarkDataset(root=args.dataset_path, npoints=args.point_num, class_choice=args.class_choice)
        self.dataLoader = tf.data.Dataset.from_tensor_slices(self.data).batch(args.batch_size).shuffle(True).repeat().prefetch(tf.data.AUTOTUNE)
        print("Training Dataset : {} prepared.".format(len(self.data)))
        
        self.G = WarpingGAN(num_points=2048)
        self.G_optimizer = tf.keras.optimizers.Adam(lr=args.lr, beta_1=0, beta_2=0.99)
        self.G.compile(loss=lambda y_true, y_pred: y_pred)

        self.D = Discriminator(batch_size=args.batch_size, features=args.D_FEAT)
        self.D_optimizer = tf.keras.optimizers.Adam(lr=args.lr, beta_1=0, beta_2=0.99)
        self.D.compile(loss='binary_crossentropy')

        self.GP = GradientPenalty(args.lambdaGP, gamma=1, device=args.device)
        print("Network prepared.")


    def run(self):
        epoch_log = 0
        loss_log = {'G_loss': [], 'D_loss': []}

        for epoch in range(epoch_log, self.args.epochs + 1):
            for _iter, data in enumerate(self.dataLoader):
                point, _ = data
                point = point.numpy()
                start_time = time.time()

                for d_iter in range(self.args.D_iter):
                    self.D.trainable = True
                    self.G.trainable = False

                    self.D_optimizer.zero_grad()

                    z = tf.random.normal((self.args.batch_size, 1, 128))

                    with tf.GradientTape() as tape:
                        fake_point = self.G(z, training=False)
                        fake_point = (fake_point)

                        D_real, real_index = self.D(point, training=True)
                        D_realm = tf.reduce_mean(D_real)
                        D_fake, _ = self.D(fake_point, training=True)
                        D_fakem = tf.reduce_mean(D_fake)

                        gp_loss = self.GP(self.D, point, fake_point)
                        d_loss = -D_realm + D_fakem
                        d_loss_gp = d_loss + gp_loss

                    d_gradients = tape.gradient(d_loss_gp, self.D.trainable_variables)
                    self.D_optimizer.apply_gradients(zip(d_gradients, self.D.trainable_variables))

                realvar = stitchloss(point, real_index)

                loss_log['D_loss'].append(d_loss.numpy())                  

                self.G.trainable = True
                self.D.trainable = False

                self.G_optimizer.zero_grad()
                z = tf.random.normal((self.args.batch_size, 1, 128))
                fake_point = self.G(z, training=True)
                fake_point = (fake_point)
                G_fake, fake_index = self.D(fake_point, training=True)
                fakevar = stitchloss(fake_point,fake_index)
                G_fakem = tf.reduce_mean(G_fake)
                varloss = tf.math.pow((fakevar-realvar),2)
                g_loss = -G_fakem + 0.05*varloss
                g_gradients = tape.gradient(g_loss, self.G.trainable_variables)
                self.G_optimizer.apply_gradients(zip(g_gradients, self.G.trainable_variables))

                loss_log['G_loss'].append(g_loss.numpy())
    
                print("[Epoch/Iter] ", "{:3} / {:3}".format(epoch, _iter),
                      "[ D_Loss ] ", "{: 7.6f}".format(d_loss), 
                      "[ G_Loss ] ", "{: 7.6f}".format(g_loss), 
                      "[ Time ] ", "{:4.2f}s".format(time.time()-start_time))

            if epoch % 50 == 0:
                tf.keras.models.save_model(self.G, str(epoch) + '.h5')

                print('Checkpoint is saved.')
                   

if __name__ == '__main__':
    args = Arguments().parser().parse_args()

    model = WarpingGANTrain(args)
    model.run()