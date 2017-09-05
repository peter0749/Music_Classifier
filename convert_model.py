import conv_net_sound
input_shape = [330491]
model_old = conv_net_sound.conv_net(input_shape=input_shape, class_n = 16)
model_old.load_weights('./top_weight.h5')
model_new = conv_net_sound.conv_net(input_shape=[131072], class_n=None)
model_old_dict = dict([(layer.name, layer) for layer in model_old.layers])
model_new_dict = dict([(layer.name, layer) for layer in model_new.layers])
for lname in model_old_dict:
    for rname in model_new_dict:
        if lname == rname:
            try:
                model_new_dict[rname].set_weights(model_old_dict[lname].get_weights())
            except:
                print lname
                continue

model_new.save_weights('./conv_net.h5')

