import os
import math
import datetime

import tensorflow as tf
from tqdm import tqdm

from model import efficientnetv2_s as create_model
from utils import generate_ds

assert tf.version.VERSION >= "2.4.0", "version of tf must greater/equal than 2.4.0"


def main():
    data_root = "/data/flower_photos"  # get data root path

    if not os.path.exists("./save_weights"):
        os.makedirs("./save_weights")

    img_size = {"s": [300, 384],  # train_size, val_size
                "m": [384, 480],
                "l": [384, 480]}
    num_model = "s"

    batch_size = 8
    epochs = 30
    num_classes = 5
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
    model.build((1, img_size[num_model][0], img_size[num_model][0], 3))

    # 下载我提前转好的预训练权重
    # 链接: https://pan.baidu.com/s/1Pr-pO5sQVySPQnBY8pQH7w  密码: f6hi
    # load weights
    '''
    mod
    '''
    # pre_weights_path = './efficientnetv2-s.h5'
    # assert os.path.exists(pre_weights_path), "cannot find {}".format(pre_weights_path)
    # model.load_weights(pre_weights_path, by_name=True, skip_mismatch=True)

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

    # Stats
    bands = list(range(len(band_names)))
    # dim = tgp.get_raster_info(df_train['fp'].values[0])[:2]
    dim = [512, 512]
    # print(f'dim = {dim}, ')
    n_channels = len(bands)
    leu_classes_ipcc3 = 2 # Leu, others
    # n_classes_ipcc2 = len(set(df_train['Y_ipcc2'].tolist() + df_test['Y_ipcc2'].tolist()))
    # n_classes_ipcc3 = len(set(df_train['Y_ipcc3'].tolist() + df_test['Y_ipcc3'].tolist()))
    print("dim", dim)
    print("n_channels", n_channels)
    # print("n_classes_ipcc2", n_classes_ipcc2)
    # print("n_classes_ipcc3", n_classes_ipcc3)


    # Generator
    test_generator = DataGenerator(df['fp_npy_new'].values, df['Y_ipcc2'].values, df['Y_ipcc3'].values,
                                   bands=self.bands, batch_size=batch_size, dim=self.dim, n_channels=self.n_channels,
                                   n_classes1=self.n_classes_ipcc2, n_classes2=self.n_classes_ipcc3,
                                   no_data_value=-999, augumentation=False, shuffle=False)

    train_generator = DataGenerator_Leu(df_train['fp_npy'].values, df_train['Leu_label'].values, bands=bands, batch_size=batch_size,
                                    dim=dim, n_channels=n_channels, n_classes1=leu_classes_ipcc3, no_data_value=-999, augumentation=True,
                                    shuffle=True)
    test_generator = DataGenerator_Leu(df_test['fp_npy'].values, df_test['Leu_label'].values, bands=bands, batch_size=batch_size,
                                    dim=dim, n_channels=n_channels, n_classes1=leu_classes_ipcc3, no_data_value=-999, augumentation=True,
                                    shuffle=True)




    csv_logger = CSVLogger(os.path.join(current_model_dir, dt_str+"-1-model_history_log.csv"), append=True)
    checkpoint = ModelCheckpoint(os.path.join(current_model_dir, dt_str+'-2-model-{epoch:03d}-{val_output1_accuracy:03f}-{val_output2_accuracy:03f}.h5'),
                                verbose=1)#, monitor='val_output1_accuracy', save_best_only=True, mode='max')

    # Training
    STEP_SIZE_TRAIN=len(train_generator)
    STEP_SIZE_VALID=len(test_generator)
    history = model.fit_generator(generator=train_generator,
                        initial_epoch=initial_epoch,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=test_generator,
                        validation_steps=STEP_SIZE_VALID,
                        callbacks=[checkpoint, csv_logger],
                        workers=8,
                        use_multiprocessing=True,
                        verbose=1,
                        epochs=epochs)

'''
下面直接取代
'''


    # custom learning rate curve
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
        train_loss.reset_states()  # clear history info
        train_accuracy.reset_states()  # clear history info
        val_loss.reset_states()  # clear history info
        val_accuracy.reset_states()  # clear history info

        # train
        train_bar = tqdm(train_ds)
        for images, labels in train_bar:
            train_step(images, labels)

            # print train process
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}, acc:{:.3f}".format(epoch + 1,
                                                                                 epochs,
                                                                                 train_loss.result(),
                                                                                 train_accuracy.result())

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
            save_name = "./save_weights/efficientnetv2.ckpt"
            model.save_weights(save_name, save_format="tf")


if __name__ == '__main__':
    main()
