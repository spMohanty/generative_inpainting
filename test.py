import argparse

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng

from inpaint_model import InpaintCAModel
import glob


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', default='', type=str,
                    help='The directory of tensorflow checkpoint.')


if __name__ == "__main__":
    # ng.get_gpus(1)
    args = parser.parse_args()

    model = InpaintCAModel()

    def load_inputs(image_path, mask_path):
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path)
        assert image.shape == mask.shape
        h, w, _ = image.shape
        grid = 8
        image = image[:h//grid*grid, :w//grid*grid, :]
        mask = mask[:h//grid*grid, :w//grid*grid, :]
        print('Shape of image: {}'.format(image.shape))
        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        input_image = np.concatenate([image, mask], axis=2)
        return input_image
    
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        ####
        #### Might need to change the placeholder shape based on the 
        #### Model you are using
        ####
        #### Can obtain this shape by using the `load_inputs` function
        #### to load and preprocess the input image and mask, and print 
        #### its shape 
        ####
        #### input = load_inputs(input_image_path, input_mask_path)
        #### print(input.shape)
        #### >>> (1, 512, 1360, 3)
        ####
        input_image_placeholder = tf.placeholder(tf.float32, shape=(1, 512, 1360, 3))
        output = model.build_server_graph(input_image_placeholder, reuse=tf.AUTO_REUSE)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            if not vname.startswith("custom_input_image"):
                var_value = tf.contrib.framework.load_variable(args.checkpoint_dir, from_name)
                assign_ops.append(tf.assign(var, var_value))
        sess.run(assign_ops)
        print('Model loaded.')

        for input_image_path in glob.glob("examples/places2/*_input.png"):
            print("====================================================")
            mask_path = input_image_path.replace("_input.png", "_mask.png")
            print(input_image_path)
            input_image_raw = load_inputs(input_image_path, mask_path)
            result = sess.run(output, feed_dict={
                input_image_placeholder: input_image_raw
            })
            print("Result Shape : ", result.shape)
            output_path = input_image_path.replace("_input.png", "_output.png")
            cv2.imwrite(output_path, result[0][:, :, ::-1])
