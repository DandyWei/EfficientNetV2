import os
import json
import glob
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from model import efficientnetv2_forest as create_model
from sklearn.metrics import f1_score, precision_score, recall_score
def Evaluation(test_labels, predict_class, average="macro"):
    f1 = f1_score(test_labels, predict_class, average=average)
    pre = precision_score(test_labels, predict_class, average=average)
    re = recall_score(test_labels, predict_class, average=average)
    return f1, pre, re

def gen_data(test_fps, bath_size=32, n=32):
    i = 0
    while i < n:
        test_fs, test_labels = [], []
        fps_iter = test_fps[i*32:(i+1)*32]
        for fps in fps_iter:
            test_fs.append(np.load(fps).astype(np.float32))
            test_labels.append(int(fps.parent.name))
        yield test_fs, test_labels
        i += 1


def main():
    num_classes = 20

    img_size = {"s": 256,
                "m": 480,
                "l": 480}
    num_model = "s"
    im_height = im_width = img_size[num_model]

    # load image
    # img = Image.open(img_path)
    # resize image
    # img = img.resize((im_width, im_height))
    # plt.imshow(img)

    # read image
    # img = np.array(img).astype(np.float32)

    # preprocess
    # img = (img / 255. - 0.5) / 0.5

    # Add the image to a batch where it's the only member.
    train_fps = [p for p in Path("D:/Datasets/Effi").glob("11/*.npy") if len(p.stem.split("_")) == 6]

    test_fps = [p for p in Path("D:/Datasets/test").glob("*/*.npy")]
    len(test_fps)

    np.load(test_fps[0]).min()
    # test_fs = [np.load(path) for path in test_fps]
    # test_fs = np.asarray(test_fs)


    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = create_model(num_classes=num_classes)

    model.build((1, im_height, im_width, 17))

    weights_path = Path("./save_weights/efficientnetv2.ckpt")

    model.load_weights(weights_path)

    # model_weight_fp = Path("D:\Thinktron\EfficientNetV2\Test11_efficientnetV2\save_weights\Fuse_2_laeyr_211201\efficientnetv2_2021_12_19_0.8595552444458008_28.ckpt")
    # model.load_weights(model_weight_fp)


    predicts = []
    labels = []
    len(train_fps) // 32
    for fs, label in gen_data(train_fps, 32,3):
        rs = model.predict(np.stack(fs))
        predict_class = np.argmax(rs, axis=1)
        predicts += list(predict_class)
        labels += label

    result = tf.keras.layers.Softmax(axis=0)(rs)
    predict_class = np.argmax(result)
    Evaluation(predicts, labels)
    predicts
    labels
    # predict = model.predict(nptest_fs)
    predict = np.asarray(np.concatenate(rs))
    # result = np.squeeze(predict)
    # _result = tf.keras.layers.Softmax()(result)
    # result.shape

    predict_class = np.argmax(rs, axis=1)

    # clsidx = 14
    """
    accuracy = ( TP + TN ) / T
    Recall (True Positive Rate) = TP / (TP + FN)
    Specificity (True Negative Rate) = TN / (FP + TN)

    FPR=FP / ( FP+TN ) = 1−TNR
    FNR=FN / ( TP+FN ) = 1−TPR

    F1 = 2 * (precision * recall) / (precision + recall)

    """


    """
    params : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """

    # GT = np.array([14] * len(predict_class))
    f1_score(test_labels, predict_class, average="macro")
    precision_score(test_labels, predict_class, average="macro")
    recall_score(test_labels, predict_class, average="macro")

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_class)],
                                                 result[predict_class])
    # plt.title(print_res)
    print(print_res)
    # plt.show()






if __name__ == '__main__':
    main()
