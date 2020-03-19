import tensorflow as tf

def optimizer(beta_1, loss_gen, loss_dis, loss_type, learning_rate_input_d, learning_rate_input_g=None, learning_rate_input_e=None, beta_2=None, clipping=None, display=True,
                gen_name='generator', dis_name='discriminator', mapping_name='mapping_', encoder_name='encoder', gpus=[0]):
    
    # Gather variables for each network system, mapping network is included in the generator
    trainable_variables = tf.trainable_variables()
    generator_variables = [variable for variable in trainable_variables if variable.name.startswith(gen_name)]
    discriminator_variables = [variable for variable in trainable_variables if variable.name.startswith(dis_name)]
    mapping_variables = [variable for variable in trainable_variables if variable.name.startswith(mapping_name)]
    encoder_variables = [variable for variable in trainable_variables if variable.name.startswith(encoder_name)]
    if len(mapping_variables) != 0:
        generator_variables.extend(mapping_variables)

    # Optimizer variable to track with optimizer is actually used.
    optimizer_print = ''

    # Handling Batch Normalization.
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        # Wasserstein distance with gradient penalty and Hinge loss.
        if ('wasserstein distance' in loss_type and 'gradient penalty' in loss_type) or ('hinge' in loss_type):
            train_discriminator = tf.train.AdamOptimizer(learning_rate_input_d, beta_1, beta_2).minimize(loss_dis, var_list=discriminator_variables)
            train_generator = tf.train.AdamOptimizer(learning_rate_input_g, beta_1, beta_2).minimize(loss_gen, var_list=generator_variables)
            optimizer_print += '%s - AdamOptimizer' % loss_type

        # Wasserstein distance loss.
        elif 'wasserstein distance' in loss_type and 'gradient penalty' not in loss_type:
            # Weight Clipping on Discriminator, this is done to ensure the Lipschitz constrain.
            train_discriminator = tf.train.AdamOptimizer(learning_rate_input_d, beta_1, beta_2).minimize(loss_dis, var_list=discriminator_variables)
            dis_weight_clipping = [value.assign(tf.clip_by_value(value, -clipping, clipping)) for value in discriminator_variables]
            train_discriminator = tf.group(*[train_discriminator, dis_weight_clipping])
            train_generator = tf.train.AdamOptimizer(learning_rate_input_g, beta_1, beta_2).minimize(loss_gen, var_list=generator_variables)
            optimizer_print += '%s - AdamOptimizer' % loss_type
        
        # Standard, Least square, and standard relativistic loss.
        elif 'standard' in loss_type or 'least square' in loss_type or 'relativistic' in loss_type:
            with tf.device('/gpu:%s' % gpus[0]):
                train_encoder = None
                train_discriminator = tf.train.AdamOptimizer(learning_rate=learning_rate_input_d, beta1=beta_1).minimize(loss_dis, var_list=discriminator_variables) 
            other_gpu = gpus[0]
            if len(gpus) > 1:
                other_gpu = gpus[1]
            with tf.device('/gpu:%s' % other_gpu):
                train_generator = tf.train.AdamOptimizer(learning_rate=learning_rate_input_g, beta1=beta_1).minimize(loss_gen, var_list=generator_variables)
                if len(encoder_variables) != 0 and learning_rate_input_e is not None: train_encoder = tf.train.AdamOptimizer(learning_rate=learning_rate_input_e, beta1=beta_1).minimize(loss_gen, var_list=encoder_variables)
            optimizer_print += '%s - AdamOptimizer' % loss_type
        
        else:
            print('Optimizer: Loss %s not defined' % loss_type)
            exit(1)

        if display:
            print('[Optimizer] Loss %s' % optimizer_print)
            print()
    
    if len(encoder_variables) != 0:
        return train_discriminator, train_generator, train_encoder
    else:
        return train_discriminator, train_generator


def optimizer_encoding(beta_1, loss_enc, loss_dis, loss_type, learning_rate_input_d, learning_rate_input_e, beta_2=None, display=True, dis_name='dis_encoding', encoder_name='encoder'):
    # Gather variables for each network system, mapping network is included in the generator
    trainable_variables = tf.trainable_variables()
    discriminator_variables = [variable for variable in trainable_variables if variable.name.startswith(dis_name)]
    encoder_variables = [variable for variable in trainable_variables if variable.name.startswith(encoder_name)]

    # Optimizer variable to track with optimizer is actually used.
    optimizer_print = ''

    # Handling Batch Normalization.
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        # Standard, Least square, and standard relativistic loss.
        if 'standard' in loss_type or 'least square' in loss_type or 'relativistic' in loss_type:
            train_discriminator_enc = tf.train.AdamOptimizer(learning_rate=learning_rate_input_d, beta1=beta_1).minimize(loss_dis, var_list=discriminator_variables) 
            train_encoder = tf.train.AdamOptimizer(learning_rate=learning_rate_input_e, beta1=beta_1).minimize(loss_enc, var_list=encoder_variables)
            optimizer_print += '%s - AdamOptimizer' % loss_type
        
        else:
            print('Optimizer: Loss Dis-Enc %s not defined' % loss_type)
            exit(1)

        if display:
            print('[Optimizer] Loss Dis-Enc %s' % optimizer_print)
            print()
    
    return train_discriminator_enc, train_encoder