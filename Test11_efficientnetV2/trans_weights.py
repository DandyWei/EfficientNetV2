from model import *


def main(ckpt_path: str,
         model_name: str,
         model: tf.keras.Model):
    var_dict = {v.name.split(':')[0] + "/.ATTRIBUTES/VARIABLE_VALUE": v for v in model.weights}

    reader = tf.train.load_checkpoint(ckpt_path)
    var_shape_map = reader.get_variable_to_shape_map()


    # print(f"var map = {var_shape_map}")
    for key, var in var_dict.items():
        # print(f"key = {key}, value = {var}")
        if 'blocks/0/norm' in key:
            key = key.replace("norm", "norm_1")
        # key_ = model_name + "/" + key
        # key_ = key_.replace("batch_normalization", "tpu_batch_normalization")
        if key in var_shape_map:
            if var_shape_map[key] != var.shape:
                msg = "shape mismatch: {}".format(key)
                print(msg)
            else:
                var.assign(reader.get_tensor(key), read_value=False)
        else:
            msg = "Not found {} in {}".format(key, ckpt_path)
            print(msg)

    print(f"save model to {model_name}.h5")
    model.save_weights(f"./{model_name}.h5")


if __name__ == '__main__':
    model = efficientnetv2_forest()
    model.build((1, 256, 256, 17))

    main(ckpt_path="save_weights/Fuse_2_laeyr_211201",
         model_name="efficientnetv2-forest",
         model=model)
