# code in 'Physics Informed Deep Learning for Transport in Porous Media. Buckley Leverett Problem'

import numpy as np 
import tensorflow as tf

# Network for QOI (Saturation) as a function of (x,t) 
def net_saturation(...):
    ... 
    mlp = tf.layers.dense(x, mlp_config.layer_size_lst[i], 
                            activation=mlp_config.activation_lst[i], 
                            kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                            name=mlp_config.main_name + ’_layer_’ + str(i), reuse=reuse)
    ...

# Network for PDE residual 
def net_pde_residual(s, x, t): 
    s_t = tf.gradients(s, t)[0] 
    s_x = tf.gradients(s, x)[0] 
    s_xx = tf.gradients(s_x, x)[0] 
    lambda_swc = 0.0 
    lambda_m = 2 
    lambda_sor = 0.0
    nu = 0.001 
    frac = tf.divide(tf.square(s - Swc), tf.square(s - lambda_swc) + tf.divide(tf.square(1 - s - 
        lambda_sor), lambda_m))
    frac_s = tf.gradients(frac,s)[0] 
    f = s_t + frac_s * s_x 
    f = tf.identity(f, name=’f_pred’) 
    return f

# Loss function 
def discriminator_loss(logits_real, logits_fake): 
    # x = logits, z = labels 
    # tf.nn.sigmoid_cross_entropy_with_logits <=> z * -log(sigmoid(x)) + (1 - z) * -log(1 - 
    #   sigmoid(x))
    dis_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real, 
        labels=tf.zeros_like(logits_real)))
    dis_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, 
        labels=tf.ones_like(logits_fake)))
    dis_loss = dis_loss_real + dis_loss_fake 
    return dis_loss

def generator_loss(logits_fake, logits_posterior, pde_residuals, w_posterior_loss, w_pde_loss): 
    # x = logits, z = labels 
    # tf.nn.sigmoid_cross_entropy_with_logits <=> z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
    gen_loss_entropy = tf.reduce_mean(logits_fake) gen_loss_posterior = tf.reduce_mean(tf.multiply((w_posterior_loss - 1.0), 
        tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_posterior, 
        labels=tf.ones_like(logits_posterior))))
    gen_loss_pde = w_pde_loss * tf.reduce_mean(tf.square(pde_residuals), name=’loss_pde_form’) 
    gen_loss = gen_loss_entropy + gen_loss_posterior + gen_loss_pde
    return gen_loss, gen_loss_entropy, gen_loss_posterior, gen_loss_pde
