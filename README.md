# EfficientNetV2
Refer to the version compiled by others and source codes

## Record

### 211122

## modify

### train
Modify :

  1. input size to 256,

  2. input from read images to read npy files
  3. tf function to py function
  ```python
  def load_image_wrapper_train(file, labels):
      return tf.py_function(process_train_info, [file, labels], [tf.float32, tf.int32])
  ```
  conver format for tf

  4. read npy file by py functions
  ```python
  def process_val_info(img_path, label):
      image = np.load(img_path.numpy())
      image = tf.cast(image, tf.float32)
      image = tf.image.resize_with_crop_or_pad(image, val_im_height, val_im_width)
      return image, label
  ```

  5. read images by tf functions
  ```python
  def process_val_info(img_path, label):
      image = tf.io.read_file(img_path)
      image = tf.image.decode_jpeg(image, channels=3)
      image = tf.cast(image, tf.float32)
      image = tf.image.resize_with_crop_or_pad(image, val_im_height, val_im_width)
      image = (image / 255. - 0.5) / 0.5
      return image, label
  ```




* tf.io.read_file cannot read files with suffix npy tif,
*support types ".jpg", ".JPG", ".jpeg", ".JPEG"


### utils
