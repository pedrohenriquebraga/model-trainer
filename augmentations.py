import tensorflow as tf

def preprocess(images, labels):
    return tf.cast(images, tf.float32), labels


def maybe_grayscale(image, probability=0.25):
    return tf.cond(
        tf.random.uniform([]) < probability,
        lambda: tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image)),
        lambda: image,
    )


def random_zoom_crop(image, img_size, min_scale=0.75, max_scale=0.95):
    image_shape = tf.shape(image)
    height = image_shape[0]
    width = image_shape[1]

    scale = tf.random.uniform([], min_scale, max_scale)
    crop_height = tf.maximum(1, tf.cast(tf.cast(height, tf.float32) * scale, tf.int32))
    crop_width = tf.maximum(1, tf.cast(tf.cast(width, tf.float32) * scale, tf.int32))
    offset_height = tf.random.uniform([], 0, height - crop_height + 1, dtype=tf.int32)
    offset_width = tf.random.uniform([], 0, width - crop_width + 1, dtype=tf.int32)

    image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, crop_height, crop_width)
    image = tf.image.resize(image, img_size)
    image.set_shape([img_size[0], img_size[1], 3])
    return image


def add_random_noise(image, probability=0.3, stddev=8.0):
    def noisy_image():
        noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=stddev, dtype=image.dtype)
        return tf.clip_by_value(image + noise, 0.0, 255.0)

    return tf.cond(tf.random.uniform([]) < probability, noisy_image, lambda: image)


def random_flips(image, horizontal_probability=0.5, vertical_probability=0.2):
    image = tf.cond(
        tf.random.uniform([]) < horizontal_probability,
        lambda: tf.image.flip_left_right(image),
        lambda: image,
    )
    image = tf.cond(
        tf.random.uniform([]) < vertical_probability,
        lambda: tf.image.flip_up_down(image),
        lambda: image,
    )
    return image


def color_jitter(image, probability=0.8):
    def jittered_image():
        normalized = tf.clip_by_value(image / 255.0, 0.0, 1.0)
        normalized = tf.image.random_brightness(normalized, max_delta=0.08)
        normalized = tf.image.random_contrast(normalized, lower=0.85, upper=1.15)
        normalized = tf.image.random_saturation(normalized, lower=0.85, upper=1.15)
        normalized = tf.image.random_hue(normalized, max_delta=0.04)
        return tf.clip_by_value(normalized * 255.0, 0.0, 255.0)

    return tf.cond(tf.random.uniform([]) < probability, jittered_image, lambda: image)


def translate_and_shear(image, img_size, probability=0.7, max_translation=0.08, max_shear=0.12):
    def transformed_image():
        height = tf.cast(tf.shape(image)[0], tf.float32)
        width = tf.cast(tf.shape(image)[1], tf.float32)
        translate_x = tf.random.uniform([], -max_translation, max_translation) * width
        translate_y = tf.random.uniform([], -max_translation, max_translation) * height
        shear_x = tf.random.uniform([], -max_shear, max_shear)
        shear_y = tf.random.uniform([], -max_shear, max_shear)

        transform = tf.stack([
            1.0, shear_x, translate_x,
            shear_y, 1.0, translate_y,
            0.0, 0.0,
        ])

        transformed = tf.raw_ops.ImageProjectiveTransformV3(
            images=tf.expand_dims(image, axis=0),
            transforms=tf.expand_dims(transform, axis=0),
            output_shape=tf.constant([img_size[0], img_size[1]], dtype=tf.int32),
            interpolation='BILINEAR',
            fill_mode='REFLECT',
            fill_value=0.0,
        )
        transformed = transformed[0]
        transformed.set_shape([img_size[0], img_size[1], 3])
        return transformed

    return tf.cond(tf.random.uniform([]) < probability, transformed_image, lambda: image)


def blur_or_sharpen(image, probability=0.6):
    def blur_image():
        blur_kernel = tf.constant(
            [[1.0, 2.0, 1.0],
             [2.0, 4.0, 2.0],
             [1.0, 2.0, 1.0]],
            dtype=image.dtype,
        ) / 16.0
        blur_kernel = tf.reshape(blur_kernel, [3, 3, 1, 1])
        blur_kernel = tf.tile(blur_kernel, [1, 1, 3, 1])
        blurred = tf.nn.depthwise_conv2d(tf.expand_dims(image, axis=0), blur_kernel, strides=[1, 1, 1, 1], padding='SAME')
        return tf.clip_by_value(blurred[0], 0.0, 255.0)

    def sharpen_image():
        sharpen_kernel = tf.constant(
            [[0.0, -1.0, 0.0],
             [-1.0, 5.0, -1.0],
             [0.0, -1.0, 0.0]],
            dtype=image.dtype,
        )
        sharpen_kernel = tf.reshape(sharpen_kernel, [3, 3, 1, 1])
        sharpen_kernel = tf.tile(sharpen_kernel, [1, 1, 3, 1])
        sharpened = tf.nn.depthwise_conv2d(tf.expand_dims(image, axis=0), sharpen_kernel, strides=[1, 1, 1, 1], padding='SAME')
        return tf.clip_by_value(sharpened[0], 0.0, 255.0)

    def blurred_or_sharpened():
        return tf.cond(tf.random.uniform([]) < 0.5, blur_image, sharpen_image)

    return tf.cond(tf.random.uniform([]) < probability, blurred_or_sharpened, lambda: image)


def random_cutout(image, probability=0.35, max_fraction=0.25):
    def cutout_image():
        height = tf.shape(image)[0]
        width = tf.shape(image)[1]
        fraction = tf.random.uniform([], 0.08, max_fraction)
        cut_height = tf.maximum(1, tf.cast(tf.cast(height, tf.float32) * fraction, tf.int32))
        cut_width = tf.maximum(1, tf.cast(tf.cast(width, tf.float32) * fraction, tf.int32))
        offset_height = tf.random.uniform([], 0, height - cut_height + 1, dtype=tf.int32)
        offset_width = tf.random.uniform([], 0, width - cut_width + 1, dtype=tf.int32)

        top = image[:offset_height]
        middle = image[offset_height:offset_height + cut_height]
        bottom = image[offset_height + cut_height:]

        left = middle[:, :offset_width]
        right = middle[:, offset_width + cut_width:]
        fill_value = tf.cast(tf.reduce_mean(image, axis=[0, 1], keepdims=True), dtype=image.dtype)
        cut_patch = tf.ones([cut_height, cut_width, 3], dtype=image.dtype) * fill_value
        patched_middle = tf.concat([left, cut_patch, right], axis=1)
        return tf.concat([top, patched_middle, bottom], axis=0)

    return tf.cond(tf.random.uniform([]) < probability, cutout_image, lambda: image)


def build_augment_single_image(img_size, rotation_layer):
    def augment_single_image(image):
        image = tf.cast(image, tf.float32)
        image = rotation_layer(tf.expand_dims(image, axis=0), training=True)[0]
        image = maybe_grayscale(image)
        image = random_flips(image)
        image = color_jitter(image)
        image = translate_and_shear(image, img_size)
        image = blur_or_sharpen(image)
        image = random_zoom_crop(image, img_size)
        image = random_cutout(image)
        image = add_random_noise(image)
        return image

    return augment_single_image


def build_augment_image(img_size, rotation_layer):
    augment_single_image = build_augment_single_image(img_size, rotation_layer)

    def augment_image(images, labels):
        images = tf.map_fn(
            augment_single_image,
            images,
            fn_output_signature=tf.TensorSpec(shape=(img_size[0], img_size[1], 3), dtype=tf.float32),
        )
        return images, labels

    return augment_image