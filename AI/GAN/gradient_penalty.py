import tensorflow as tf

class GradientPenalty:
    def __init__(self, lambdaGP, gamma=1, vertex_num=2500):
        self.lambdaGP = lambdaGP
        self.gamma = gamma
        self.vertex_num = vertex_num

    def __call__(self, netD, real_data, fake_data):
        batch_size = tf.shape(real_data)[0]

        fake_data = fake_data[:batch_size]

        alpha = tf.random.uniform(shape=[batch_size, 1, 1], minval=0.0, maxval=1.0)

        interpolates = real_data + alpha * (fake_data - real_data)

        with tf.GradientTape() as tape:
            tape.watch(interpolates)
            disc_interpolates, _ = netD(interpolates)

        gradients = tape.gradient(disc_interpolates, interpolates)
        gradients = tf.reshape(gradients, [batch_size, -1])
        
        gradient_penalty = (((tf.norm(gradients, ord=2, axis=1) - self.gamma) / self.gamma) ** 2).mean() * self.lambdaGP

        return gradient_penalty