import os
import math
import datetime
import glob
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
from datetime import datetime as dt
from model import efficientnetv2_forest as create_model
from utils import generate_ds
from pathlib import Path
# Path("Test11_efficientnetV2/save_weights/Fuse_2_laeyr_211201/efficientnetv2_2021_12_2_18_9_0.9856745004653931.h5")
# ps = [p for p in Path("save_weights/Fuse_2_laeyr_211201").glob("*.h5")]

assert tf.version.VERSION >= "2.4.0", "version of tf must greater/equal than 2.4.0"


def main():
    # data_root = r"/data/flower_photos"  # get data root path
    # data_root = r"D:/Thinktron/EfficientNetV2/Data/Train"
    # data_root = r"D:/Datasets/Effi"
    data_root = r"D:/Datasets/Effi"
    if not os.path.exists("./save_weights"):
        os.makedirs("./save_weights")


    model_path = Path("save_weights") / Path('Fuse_2_laeyr_211201')
    os.mkdir(model_path) if not os.path.isdir(model_path) else None

    img_size = {"s": [256, 256],  # train_size, val_size
                "m": [384, 480],
                "l": [384, 480]}
    num_model = "s"

    batch_size = 32
    epochs = 30
    num_classes = 20
    '''
    mod
    '''
    freeze_layers = True
    initial_lr = 0.01

    log_dir = "./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_writer = tf.summary.create_file_writer(os.path.join(log_dir, "train"))
    val_writer = tf.summary.create_file_writer(os.path.join(log_dir, "val"))

    # data generator with data augmentation
    train_ds, val_ds = generate_ds(data_root,
                                   train_im_height=img_size[num_model][0],
                                   train_im_width=img_size[num_model][0],
                                   val_im_height=img_size[num_model][1],
                                   val_im_width=img_size[num_model][1],
                                   batch_size=batch_size,
                                   val_rate=0.5)

    # create model
    model = create_model(num_classes=num_classes)
    '''
    b, H, W, C 這裡一定要BUILD不然會報錯
    '''
    model.build((1, img_size[num_model][0], img_size[num_model][0], 17))

    '''
    mod
    '''

    # try:
        #D:\Thinktron\EfficientNetV2\Test11_efficientnetV2\save_weights\Fuse_2_laeyr_211201\efficientnetv2_2021_12_8_8_40_0.9730437994003296.h5

    model_weight_fp = Path("D:\Thinktron\EfficientNetV2\Test11_efficientnetV2\save_weights\Fuse_2_laeyr_211201\efficientnetv2_2021_12_14_0.8457319736480713_4.ckpt")
    # model.load_weights(model_weight_fp)# , by_name=True, skip_mismatch=True
    # print(f"succese load from model path {model_weight_fp}")


    # except:
    #     print("no such a weight")
    # freeze bottom layers
    '''
    no
    '''
    # if freeze_layers:
    #     unfreeze_layers = "head"
    #     for layer in model.layers:
    #         if unfreeze_layers not in layer.name:
    #             layer.trainable = False
    #         else:
    #             print("training {}".format(layer.name))

    model.summary()

    '''
    下面直接取代
    '''

    #custom learning rate curve
    def scheduler(now_epoch):
        end_lr_rate = 0.01  # end_lr = initial_lr * end_lr_rate
        rate = ((1 + math.cos(now_epoch * math.pi / epochs)) / 2) * (1 - end_lr_rate) + end_lr_rate  # cosine
        new_lr = rate * initial_lr

        # writing lr into tensorboard
        with train_writer.as_default():
            tf.summary.scalar('learning rate', data=new_lr, step=epoch)

        return new_lr

    # using keras low level api for training
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate=initial_lr, momentum=0.9)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

    @tf.function
    def train_step(train_images, train_labels):
        with tf.GradientTape() as tape:
            output = model(train_images, training=True)
            loss = loss_object(train_labels, output)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(train_labels, output)

    @tf.function
    def val_step(val_images, val_labels):
        output = model(val_images, training=False)
        loss = loss_object(val_labels, output)

        val_loss(loss)
        val_accuracy(val_labels, output)

    best_val_acc = 0.
    c = 0
    for epoch in range(epochs):
        train_loss.reset_states()  # clear history info
        train_accuracy.reset_states()  # clear history info
        val_loss.reset_states()  # clear history info
        val_accuracy.reset_states()  # clear history info



        # train
        train_bar = tqdm(train_ds)
        val_bar = tqdm(val_ds)

        # l1 = [[1, 2], [2, 3]]
        # l2 = [[3, 5], [5, 6]]
        # for (i, j), (k, l) in zip(l1, l2):
        #     print(i, j, k, l)

        for (train_images, train_labels), (val_images, val_labels) in zip(train_bar, val_bar):
            c += 1

            train_step(train_images, train_labels)

            # print train process
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}, acc:{:.3f}".format(epoch + 1,
                                                                                 epochs,
                                                                                 train_loss.result(),
                                                                                 train_accuracy.result())

            val_step(val_images, val_labels)

            val_bar.desc = "valid epoch[{}/{}] loss:{:.3f}, acc:{:.3f}".format(epoch + 1,
                                                                               epochs,
                                                                               val_loss.result(),
                                                                               val_accuracy.result())

        # update learning rate
        optimizer.learning_rate = scheduler(epoch)

        # validate
        # val_bar = tqdm(val_ds)
        # for images, labels in val_bar:
        #     val_step(images, labels)
        #
        #     # print val process
            # val_bar.desc = "valid epoch[{}/{}] loss:{:.3f}, acc:{:.3f}".format(epoch + 1,
            #                                                                    epochs,
            #                                                                    val_loss.result(),
            #                                                                    val_accuracy.result())


        # writing training loss and acc
        with train_writer.as_default():
            tf.summary.scalar("loss", train_loss.result(), epoch)
            tf.summary.scalar("accuracy", train_accuracy.result(), epoch)

        # writing validation loss and acc
        with val_writer.as_default():
            tf.summary.scalar("loss", val_loss.result(), epoch)
            tf.summary.scalar("accuracy", val_accuracy.result(), epoch)

        # only save best weights
        # if val_accuracy.result() > best_val_acc:
        best_val_acc = val_accuracy.result()
        now = dt.now()
        # print(f"now is {now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}")
        save_name = f"./{model_path}/efficientnetv2_{now.year}_{now.month}_{now.day}_{best_val_acc}_{epoch}.ckpt"
        # model.save_weights(save_name, save_format="tf")
        model.save_weights(save_name, save_format="tf")


if __name__ == '__main__':
    main()
