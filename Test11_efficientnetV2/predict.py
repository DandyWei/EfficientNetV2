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
    train_fps = [p for p in Path("D:/Datasets/Effi").glob("14/*.npy") if len(p.stem.split("_")) == 6]

    test_fps = [p for p in Path("D:/Datasets/test").glob("*/*.npy")]
    len(test_fps)
    # test_fs = [np.load(path) for path in test_fps]
    # test_fs = np.asarray(test_fs)


    test_fs, test_labels = [], []
    for fps in test_fps:
        test_fs.append(np.load(fps))
        test_labels.append(int(fps.parent.name))
    # test_fs = np.concatenate(test_fs)
    # img.shape
    # img = (np.expand_dims(img, 0))

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
    # model_weight_fp = Path("D:/Thinktron/EfficientNetV2/Test11_efficientnetV2/save_weights/Fuse_2_laeyr_211201/efficientnetv2_2021_12_19_0.8595552444458008_28.ckpt.index")
    # assert len([p for p in model_path.glob("*")]), f"cannot find {weights_path}"
    model.load_weights(weights_path)

    nptest_fs = np.asarray(test_fs, dtype=np.ndarray)

    prrdicts = []
    batch_size = 32
    from tqdm import tqdm
    np.concatenate(test_fs[::3]).shape
    model.predict()
    for i in tqdm(range(int(np.ceil(len(nptest_fs) / batch_size)))):
        concat = np.concatenate(nptest_fs[ batch_size * i: batch_size * (i + 1) ])
        rs = model.predict(concat)
        prrdicts.append(rs)

    # predict = model.predict(nptest_fs)
    predict = np.asarray(np.concatenate(prrdicts))
    # result = np.squeeze(predict)
    # _result = tf.keras.layers.Softmax()(result)
    # result.shape

    predict_class = np.argmax(predict, axis=1)

    # clsidx = 14
    """
    accuracy = ( TP + TN ) / T
    Recall (True Positive Rate) = TP / (TP + FN)
    Specificity (True Negative Rate) = TN / (FP + TN)

    FPR=FP / ( FP+TN ) = 1−TNR
    FNR=FN / ( TP+FN ) = 1−TPR

    F1 = 2 * (precision * recall) / (precision + recall)

    """
    accuracy = sum([True for i in predict_class if i == clsidx]) / len(predict_class)


    """
    params : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    from sklearn.metrics import f1_score, precision_score, recall_score
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
