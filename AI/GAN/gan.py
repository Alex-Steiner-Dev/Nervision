import tensorflow as tf 
import time
import tensorlayer as tl
import numpy as np
import random 
import argparse

parser = argparse.ArgumentParser(description='3D-GAN implementation for 32*32*32 voxel output')

args = parser.parse_args()

checkpoint_dir = "checkpoint/"
save_dir =  "savepoint/"
output_size = 32 

real_models = tf.placeholder(tf.float32, [args.batchsize, output_size, output_size, output_size] , name='real_models')
z           = tf.random_normal((args.batchsize, 200), 0, 1)
a = tf.Print(z, [z], message="This is a: ")

net_g , G_train = generator_32(z, is_train=True, reuse = False, sig= True, batch_size=args.batchsize)
dis = discriminator

net_d , D_fake      = dis(G_train, output_size, batch_size= args.batchsize, sig = True, is_train = True, reuse = False)
net_d2, D_legit     = dis(real_models,  output_size, batch_size= args.batchsize, sig = True, is_train= True, reuse = True)
net_d2, D_eval      = dis(real_models,  output_size, batch_size= args.batchsize, sig = True, is_train= False, reuse = True) # this is for desciding weather to train the discriminator

d_loss = -tf.reduce_mean(tf.log(D_legit) + tf.log(1. - D_fake))
g_loss = -tf.reduce_mean(tf.log(D_fake))


g_vars = net_g.all_params   
d_vars = net_d.all_params  

g_vars = tl.layers.get_variables_with_name('gen', True, True)   
d_vars = tl.layers.get_variables_with_name('dis', True, True)

d_optim = tf.train.AdamOptimizer(args.discriminator_learning_rate, beta1=0.5).minimize(d_loss, var_list=d_vars)
g_optim = tf.train.AdamOptimizer(args.genorator_learning_rate, beta1=0.5).minimize(g_loss, var_list=g_vars)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session()
tl.ops.set_gpu_fraction(sess=sess, gpu_fraction=0.998)
sess.run(tf.global_variables_initializer())

files,iter_counter = grab_files(args.data) 
Train_Dis = True 
if len(args.load_epoch)>1: 
    start = int(args.load_epoch)
else: 
    start = 0 
for epoch in range(start, args.epochs):
    random.shuffle(files)
    for idx in range(len(files)/args.batchsize):
        file_batch = files[idx*args.batchsize:(idx+1)*args.batchsize]
        models, start_time = make_inputs(file_batch)
        if Train_Dis: 
            errD,_,ones = sess.run([d_loss, d_optim, D_legit] ,feed_dict={real_models: models}) 
        else: 
            ones = sess.run([D_eval] ,feed_dict={real_models: models}) 
        errG,_,zeros,objects = sess.run([g_loss, g_optim, D_fake, G_train], feed_dict={})    
        Train_Dis = (cal_acc(zeros,ones)<0.95) # only train discriminator at certain level of accuracy 
       
        print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, args.epochs, idx, len(files)/args.batchsize , time.time() - start_time, errD, errG))
    #saving generated objects
    if np.mod(epoch, args.sample ) == 0:     
        print(objects)
    