import os
import json
import glob
import numpy as np

from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from model import efficientnetv2_forest as create_model


def main():
    num_classes = 20

    img_size = {"s": 256,
                "m": 480,
                "l": 480}
    num_model = "s"
    im_height = im_width = img_size[num_model]

    # load image

    img_path = "D:/Thinktron/EfficientNetV2/Data/Test/0/190128f_57_0043_000_000_000.npy"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = np.load(img_path)
    # img = Image.open(img_path)
    # resize image
    # img = img.resize((im_width, im_height))
    # plt.imshow(img)

    # read image
    # img = np.array(img).astype(np.float32)

    # preprocess
    # img = (img / 255. - 0.5) / 0.5

    # Add the image to a batch where it's the only member.
    img = (np.expand_dims(img, 0))

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = create_model(num_classes=num_classes)

    model.build((1, im_height, im_width, 17))
    # weights_path = './save_weights/efficientnetv2.ckpt'
    from pathlib import Path
    # model_path = Path("save_weights") / Path('Fuse_2_laeyr_211201')
    weights_path = Path("efficientnetv2-forest.h5")
    # assert len([p for p in model_path.glob("*")]), f"cannot find {weights_path}"
    model.load_weights(weights_path)

    result = np.squeeze(model.predict(img))
    result = tf.keras.layers.Softmax()(result)
    predict_class = np.argmax(result)

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_class)],
                                                 result[predict_class])
    # plt.title(print_res)
    print(print_res)
    # plt.show()


if __name__ == '__main__':
    main()
