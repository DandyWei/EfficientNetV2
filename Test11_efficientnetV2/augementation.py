
from random import choice
from numpy import deg2rad, flipud, fliplr
from numpy.random import uniform, random_integers
from skimage.transform import AffineTransform, SimilarityTransform, warp

def augmentation_image(image, rotation_range=20, shear_range=0.2, scale_range=1.2, transform_range=0.,
                          horizontal_flip=True, vertical_flip=True, warp_mode='constant', cval=0):

    image_shape = image.shape

    # Generate image transformation parameters
    rotation_angle = uniform(low=-abs(rotation_range), high=abs(rotation_range))
    shear_angle = uniform(low=-abs(shear_range), high=abs(shear_range))
    scale_value = uniform(low=abs(1 / scale_range), high=abs(scale_range))
    translation_values = (random_integers(-abs(transform_range), abs(transform_range)),
                          random_integers(-abs(transform_range), abs(transform_range)))

    # Horizontal and vertical flips
    if horizontal_flip:
        # randomly flip image up/down
        if choice([True, False]):
            image = flipud(image)
    if vertical_flip:
        # randomly flip image left/right
        if choice([True, False]):
            image = fliplr(image)

    # Generate transformation object
    transform_toorigin = SimilarityTransform(scale=(1, 1), rotation=0, translation=(-image_shape[0], -image_shape[1]))
    transform_revert = SimilarityTransform(scale=(1, 1), rotation=0, translation=(image_shape[0], image_shape[1]))
    transform = AffineTransform(scale=(scale_value, scale_value), rotation=deg2rad(rotation_angle),
                                shear=deg2rad(shear_angle), translation=translation_values)
    # Apply transform
    image = warp(image, ((transform_toorigin) + transform) + transform_revert, mode=warp_mode, preserve_range=True)
    return image


if __name__ == "__main__":
    import os
    import numpy as np
    import TronGisPy as tgp
    import matplotlib.pyplot as plt

    # project_dir = r'U:'
    # training_dir = os.path.join(project_dir, 'GR2002_森林AI判釋', '07_GIS 資料', 'CodeGeneration', 'training_3cls')
    training_dir = r'U:\GR2101_110年森林AI\03_GIS資料\CodeGeneration\training_3cls'
    training_dirs = sorted([os.path.join(training_dir, d) for d in os.listdir(training_dir)])
    training_dirs_mapping = dict([(int(os.path.split(t_d)[-1]), t_d) for t_d in training_dirs])

    tree_type_order = [10, 6, 14, 16, 9]
    # tree_type_order = [10, 6, 9, 22, 4, 26, 16, 7, 13, 14, 19]
    for d_idx, tree_type_id in enumerate(tree_type_order):
        d = training_dirs_mapping.get(tree_type_id)
        fps = sorted([os.path.join(d, f) for f in os.listdir(d)])

        for idx, fp in enumerate(np.random.choice(fps, size=2, replace=False)):
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 4))
            x = tgp.get_raster_data(fp)
            x[x==-999] = 0
            x_tran = augmentation_image(x)
            ax1.imshow(tgp.Normalizer().fit_transform(x[:, :, :3]))
            ax1.set_title("original rgb")
            ax2.imshow(tgp.Normalizer().fit_transform(x[:, :, 3]), cmap='gray')
            ax2.set_title("original nir")
            ax3.imshow(tgp.Normalizer().fit_transform(x_tran[:, :, :3]))
            ax3.set_title("augumented rgb")
            ax4.imshow(tgp.Normalizer().fit_transform(x_tran[:, :, 3]), cmap='gray')
            ax4.set_title("augumented nir")

        #     fig_fp = os.path.join(project_dir, 'GR2002_森林AI判釋', '07_GIS 資料', '中間產物', '08. data_augumentation', 'Final', str(idx)+'.png')
        #     plt.savefig(fig_fp)
            plt.show()
