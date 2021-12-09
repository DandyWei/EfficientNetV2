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
    model.save_weights("./{}.h5".format("test100"))


if __name__ == '__main__':
    model = efficientnetv2_forest()
    model.build((1, 256, 256, 17))
    # var_dict = {v.name.split(':')[0] + "/.ATTRIBUTES/VARIABLE_VALUE": v for v in model.weights}
    # len(var_dict.keys())
    # for key in var_dict.keys():
    #     if "blocks/0" in key:
    #         print(key)
    # for key in var_shape_map.keys():
    #     if "blocks/0" in key:
    #         print(key)
    # reader = tf.train.load_checkpoint("save_weights/Fuse_2_laeyr_211201")
    # var_shape_map = reader.get_variable_to_shape_map()
    # len(var_shape_map)

    # ckpt_path="save_weights/Fuse_2_laeyr_211201"
    # model_name="efficientnetv2-forest"
    # 'blocks/0/norm_1/gamma/.ATTRIBUTES/VARIABLE_VALUE' in var_shape_map
    # var_dict.keys()

    # for key, var in var_dict.items():
    #     # print(f"key = {key}, value = {var}")
    #     if 'blocks/0/norm' in key:
    #         key = key.replace("norm", "norm_1")
    #     # key_ = model_name + "/" + key
    #     # key_ = key_.replace("batch_normalization", "tpu_batch_normalization")
    #     if key in var_shape_map:
    #         if var_shape_map[key] != var.shape:
    #             msg = "shape mismatch: {}".format(key)
    #             print(msg)
    #         else:
    #             var.assign(reader.get_tensor(key), read_value=False)
    #     else:
    #         msg = "Not found {} in {}".format(key, ckpt_path)
    #         print(msg)

    #D:\Thinktron\EfficientNetV2\Test11_efficientnetV2\save_weights\efficientnetv2.index
    #D:\Thinktron\EfficientNetV2\Test11_efficientnetV2\save_weights\Fuse_2_laeyr_211201\efficientnetv2_2021_12_2_18_9_0.9856745004653931.h5
    main(ckpt_path="save_weights/Fuse_2_laeyr_211201",
         model_name="efficientnetv2-forest",
         model=model)

    # model = efficientnetv2_m()
    # model.build((1, 224, 224, 3))
    # main(ckpt_path="./efficientnetv2-m-21k-ft1k/model",
    #      model_name="efficientnetv2-m",
    #      model=model)

    # model = efficientnetv2_l()
    # model.build((1, 224, 224, 3))
    # main(ckpt_path="./efficientnetv2-l-21k-ft1k/model",
    #      model_name="efficientnetv2-l",
    #      model=model)
