import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
import seaborn as sns
from gan_util import *

sns.set(color_codes=True)

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)
latent_dim = 5


class GeneratorDistribution(object):
    def __init__(self, ranges):
        self.ranges = ranges

    def sample_1(self, N):
        sample_s = np.linspace(-self.ranges, self.ranges, N) + \
                   np.random.random(N) * 0.01
        return sample_s

    def sample(self, N, dim):
        sample_s = np.random.uniform(low=-self.ranges, high=self.ranges, size=(N, dim))
        return sample_s

class GAN:

    def __init__(self, args):
        self.args = args
        self.batch_size = args.batch_size
        self.num_steps = args.num_steps
        self.log_every = args.log_every
        self.anim_path = args.anim_path
        self.anim_every = args.anim_every
        self.dim = args.dim
        self.model = None
        return

    @staticmethod
    def linear(inputs, output_dim, scope=None, stddev=1.0):
        with tf.variable_scope(scope):   # or 'linear' , reuse=tf.AUTO_REUSE
            w = tf.get_variable(
                'w',
                [inputs.get_shape()[1], output_dim],
                initializer=tf.random_normal_initializer(stddev=stddev)
            )

            b = tf.get_variable(
                'b',
                [output_dim],
                initializer=tf.constant_initializer(0.0)
            )
            return tf.matmul(inputs, w) + b

    @staticmethod
    def generator(inputs, h_dim, output_dim):
        h0 = tf.nn.softplus(GAN.linear(inputs, h_dim, 'g0'))
        print(h0.shape)
        h1 = GAN.linear(h0, output_dim, 'g1')
        print(h1.shape)
        return h1

    @staticmethod
    def discriminator(inputs, h_dim, minibatch_layer=True):
        h0 = tf.nn.relu(GAN.linear(inputs, h_dim * 2, 'd0'))
        print(h0.shape)
        h1 = tf.nn.relu(GAN.linear(h0, h_dim * 2, 'd1'))
        print(h1.shape)

        # without the minibatch layer, the discriminator needs an additional layer
        # to have enough capacity to separate the two distributions correctly
        if minibatch_layer:
            h2 = GAN.minibatch(h1)
        else:
            h2 = tf.nn.relu(GAN.linear(h1, h_dim * 2, scope='d2'))

        h3 = tf.sigmoid(GAN.linear(h2, 1, scope='d3'))
        return h3

    @staticmethod
    def minibatch(inputs, num_kernels=5, kernel_dim=3):
        x = GAN.linear(inputs, num_kernels * kernel_dim, scope='minibatch', stddev=0.02)
        activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
        diffs = tf.expand_dims(activation, 3) - \
                tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
        abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
        minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
        return tf.concat([inputs, minibatch_features], 1)

    @staticmethod
    def optimizer(loss, var_list):
        learning_rate = 0.001
        step = tf.Variable(0, trainable=False)
        optimizers = tf.train.AdamOptimizer(learning_rate).minimize(
            loss,
            global_step=step,
            var_list=var_list
        )
        return optimizers

    @staticmethod
    def log(x):
        return tf.log(tf.maximum(x, 1e-5))

    class Model:
        def __init__(self, params):

            # self.x = None
            with tf.variable_scope('G'):
                self.z = tf.placeholder(tf.float32, shape=(None, latent_dim))
                self.G = GAN.generator(self.z, params.hidden_size, params.dim)

            self.x = tf.placeholder(tf.float32, shape=(None, params.dim))
            with tf.variable_scope('D'):
                self.D1 = GAN.discriminator(self.x, params.hidden_size, params.minibatch)
            with tf.variable_scope('D', reuse=True):
                self.D2 = GAN.discriminator(self.G, params.hidden_size, params.minibatch)

            # Define the loss for discriminator and generator networks
            self.loss_d = tf.reduce_mean(-GAN.log(self.D1) - GAN.log(1 - self.D2))
            self.loss_g = tf.reduce_mean(-GAN.log(self.D2))

            var_s = tf.trainable_variables()
            self.d_params = [v for v in var_s if v.name.startswith('D/')]
            self.g_params = [v for v in var_s if v.name.startswith('G/')]

            self.opt_d = GAN.optimizer(self.loss_d, self.d_params)
            self.opt_g = GAN.optimizer(self.loss_g, self.g_params)

    def train(self, data):
        self.model = self.Model(self.args)
        gen = GeneratorDistribution(ranges=8)
        loss_dirscriminator = []
        loss_generator = []

        # tf.reset_default_graph()
        with tf.Session() as session:
            tf.local_variables_initializer().run()
            tf.global_variables_initializer().run()
            for step in range(self.num_steps + 1):
                # update discriminator
                x = random_sample(data, self.batch_size)
                z = gen.sample(self.batch_size, latent_dim)
                loss_d, _, d1, d2 = session.run([self.model.loss_d, self.model.opt_d, self.model.D1, self.model.D2], {
                    self.model.x: np.reshape(x, (self.batch_size, self.dim)),
                    self.model.z: np.reshape(z, (self.batch_size, latent_dim))
                })

                # update generator
                z = gen.sample(self.batch_size, latent_dim)
                loss_g, _ = session.run([self.model.loss_g, self.model.opt_g], {
                    self.model.z: np.reshape(z, (self.batch_size, latent_dim))
                })


                loss_dirscriminator.append(loss_d)
                loss_generator.append(loss_g)

                if step % self.log_every == 0:
                    print('{}: {:.4f}\t{:.4f}'.format(step, loss_d, loss_g))
                    # print(np.average(d1), np.average(d2))
                    print()

            plot_loss(loss_dirscriminator, loss_generator, self.args.file_name)  # plot loss

            # samps = samples(self.model, session, data, gen.ranges, self.batch_size)
            # plot_distributions(samps, gen.ranges)  # plot data's distributionss


    def get_prob(self, inputs):
        with tf.Session() as session:
            tf.local_variables_initializer().run()
            tf.global_variables_initializer().run()
            n = len(inputs)
            gen = GeneratorDistribution(ranges=8)
            z = gen.sample(n, latent_dim)
            D1, D2 = session.run([self.model.D1, self.model.D2], {  # , model.opt_d
                        self.model.x: np.reshape(inputs, (n, self.dim)),
                        self.model.z: np.reshape(z, (n, latent_dim))
                })
        return D1, D2


    def get_gen_out(self, n):
        with tf.Session() as session:
            tf.local_variables_initializer().run()
            tf.global_variables_initializer().run()
            gen = GeneratorDistribution(ranges=8)
            z = gen.sample(n, latent_dim)
            G = session.run([self.model.G], {self.model.z: np.reshape(z, (n, latent_dim))})
        return G