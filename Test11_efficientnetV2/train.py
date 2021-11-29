import os
import math
import datetime
import glob
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
from datetime import datetime as dt
from model import efficientnetv2_s as create_model
from utils import generate_ds

assert tf.version.VERSION >= "2.4.0", "version of tf must greater/equal than 2.4.0"


def main():
    # data_root = r"/data/flower_photos"  # get data root path
    # data_root = r"D:/Thinktron/EfficientNetV2/Data/Train"
    data_root = r"D:/Datasets/Effi"
    if not os.path.exists("./save_weights"):
        os.makedirs("./save_weights")

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
                                   batch_size=batch_size)

    # create model
    model = create_model(num_classes=num_classes)
    '''
    b, H, W, C
    '''
    model.build((1, img_size[num_model][0], img_size[num_model][0], 17))

    # 下载我提前转好的预训练权重
    # 链接: https://pan.baidu.com/s/1Pr-pO5sQVySPQnBY8pQH7w  密码: f6hi
    # load weights
    '''
    mod
    '''
    # pre_weights_path = './efficientnetv2-s.h5'
    # assert os.path.exists(pre_weights_path), "cannot find {}".format(pre_weights_path)
    # last_model_path = 'save_weights/'
    # model.load_weights(last_model_path, by_name=True, skip_mismatch=True)
    try:
        # weights_path = './save_weights/efficientnetv2'
        # assert len(glob.glob(weights_path+"*")), "cannot find {}".format(weights_path)
        model.load_weights("my_model.h5")
        print(f"succese load from model path {weights_path}")
    except:
        print("no such a weight")
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
    for epoch in range(epochs):
        # train_loss.reset_states()  # clear history info
        # train_accuracy.reset_states()  # clear history info
        # val_loss.reset_states()  # clear history info
        # val_accuracy.reset_states()  # clear history info



        # train
        train_bar = tqdm(train_ds)
        c = 0
        for images, labels in train_bar:
            train_step(images, labels)

            # print train process
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}, acc:{:.3f}".format(epoch + 1,
                                                                                 epochs,
                                                                                 train_loss.result(),
                                                                                 train_accuracy.result())


            c += 1
            print(c)
            if c == 7:
                model.save_weights('my_model.h5')
                exit(-1)

        # update learning rate
        optimizer.learning_rate = scheduler(epoch)

        # validate
        val_bar = tqdm(val_ds)
        for images, labels in val_bar:
            val_step(images, labels)

            # print val process
            val_bar.desc = "valid epoch[{}/{}] loss:{:.3f}, acc:{:.3f}".format(epoch + 1,
                                                                               epochs,
                                                                               val_loss.result(),
                                                                               val_accuracy.result())


        # writing training loss and acc
        with train_writer.as_default():
            tf.summary.scalar("loss", train_loss.result(), epoch)
            tf.summary.scalar("accuracy", train_accuracy.result(), epoch)

        # writing validation loss and acc
        with val_writer.as_default():
            tf.summary.scalar("loss", val_loss.result(), epoch)
            tf.summary.scalar("accuracy", val_accuracy.result(), epoch)

        # only save best weights
        if val_accuracy.result() > best_val_acc:
            best_val_acc = val_accuracy.result()
            now = dt.now()
            # print(f"now is {now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}")
            save_name = f"./save_weights/efficientnetv2_{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{best_val_acc}.h5"
            # model.save_weights(save_name, save_format="tf")
            model.save_weights(save_name)


if __name__ == '__main__':
    main()
