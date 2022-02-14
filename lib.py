import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, LeakyReLU, Add
from tensorflow.keras.optimizers import Adam
import tqdm
import numpy as np
import cv2
import base64
import tempfile
import math
import logging
import random
from const import TEMPLATE

GLOBAL_RAND_SEED = 1314
tf.random.set_seed(GLOBAL_RAND_SEED)
np.random.seed(GLOBAL_RAND_SEED)
random.seed(GLOBAL_RAND_SEED)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def set_gpu(gpu_idx):
    gpus = tf.config.list_physical_devices('GPU')
    
    print(gpus)

    if gpu_idx < 0:
        tf.config.experimental.set_visible_devices([], 'GPU')
        logging.info("Runing on CPU mode.")
    elif gpu_idx >= len(gpus):
        logging.warning("Failed to access GPU:{}. Runing on CPU mode".format(gpu_idx))
        tf.config.set_visible_devices([], 'GPU')
    else:
        tf.config.set_visible_devices(gpus[gpu_idx], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[gpu_idx], True)
        logging.info("Runing on GPU:{}".format(gpu_idx))
        # logical_gpus = tf.config.list_logical_devices('GPU')
        # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")

class FitTechA2DataDenerator:
    def __init__(self, image_paths=None, batch_size=1, save_folder=".", max_memory_size_GB=2, min_match_score=0.4, train_ratio = 0.9):
        self.prepare_template()
        self.batch_size = batch_size
        self.min_match_score = min_match_score
        self.image_num = 0

        if image_paths is None:
            return
        
        self.fp = tempfile.SpooledTemporaryFile(max_size=int(max_memory_size_GB*(1024**3)), dir=save_folder)
        self.fp_offsets = [0]
        
        logging.info("Loading training data")
        progress_bar = tqdm.tqdm(image_paths, dynamic_ncols=True, bar_format="Progress:{percentage:3.0f}% [{n_fmt}/{total_fmt}] [{elapsed}<{remaining}]")
        for path in progress_bar:
            image = cv2.imread(path, cv2.IMREAD_COLOR)

            if image is None:
                print("\r",end="")
                logging.warning("Failed to read the image. Skip: {}".format(path))
                continue

            image, score = self.template_match(image)
            if score < self.min_match_score:
                print("\r",end="")
                logging.warning("Failed to match the template. Skip: {}".format(path))
                continue

            image_shape = np.array(image.shape,"uint16")

            image_shape = base64.b64encode(image_shape)
            self.fp.write(image_shape)
            self.fp_offsets.append(self.fp.tell())

            image = base64.b64encode(image)
            self.fp.write(image)
            self.fp_offsets.append(self.fp.tell())

            self.image_num += 1
        
        less_num = batch_size * int(math.floor(1/(1-train_ratio)))
        if self.image_num < less_num:
            raise Exception("Needs {} valid images for training at least.".format(less_num))
        
        self.fp.seek(0)
        self.train_idx = []
        self.valid_idx = []

        rand_idx =  np.random.permutation(self.image_num)
        train_num = int(math.ceil(train_ratio*self.image_num))
        self.train_idx =rand_idx[:train_num]
        self.valid_idx =rand_idx[train_num:]
        
    def get_batch_data(self, training):

        batch_images = []

        if training:
            batch_idx = np.random.permutation(self.train_idx)[:self.batch_size]
        else:
            batch_idx = np.random.permutation(self.valid_idx)[:self.batch_size]

        for i in batch_idx:
            image = self.get_image_index(i)
            batch_images.append(image)

        batch_images = np.concatenate(batch_images, axis=0)
        return batch_images
    
    def get_image_index(self, idx):

        if idx < 0 and idx >= self.image_num:
            raise Exception("Out of index.")

        self.fp.seek(self.fp_offsets[idx*2])
        image_shape = self.fp.readline(self.fp_offsets[idx*2+1]-self.fp_offsets[idx*2])
        image_shape = base64.b64decode(image_shape)
        image_shape = np.frombuffer(image_shape,"uint16")
        
        self.fp.seek(self.fp_offsets[idx*2+1])
        image = self.fp.readline(self.fp_offsets[idx*2+2]-self.fp_offsets[idx*2+1])
        image = base64.b64decode(image)
        image = np.frombuffer(image,"uint8")
        image = np.reshape(image, image_shape)

        image = np.reshape(image, (1,)+image.shape[:2]+(-1,))
        return image

    def prepare_template(self):
        # ======== template from "FitTech_LD\\raw\\00031.png" ========
        # template = cv2.imread("fittech_template.png", cv2.IMREAD_COLOR)
        # print(template.shape)
        # fp = open("image.txt","a")
        # fp.write(base64.b64encode(template).decode("utf-8"))
        # fp.close()

        # the template include all regions of FitTech LD
        self.template = TEMPLATE
        self.template = base64.b64decode(self.template)
        self.template = np.frombuffer(self.template,"uint8")
        self.template = np.reshape(self.template, (841, 1051, 3))

        # the mask of FitTech LD A2 region
        self.crop_size = (512,512)
        self.crop_center = (425,335)
        self.mask = np.zeros(self.crop_size + (1, ),"uint8")
        self.mask = cv2.circle(self.mask,(256,256), 208, 1, -1)
        self.mask = cv2.rectangle(self.mask,(256,144),(480,364),1,-1)
    
    def template_match(self, image):
        if image.shape[0] < self.template.shape[0] or image.shape[1] < self.template.shape[1]:
            raise Exception("The image size is less than template (1051, 841).")

        res = cv2.matchTemplate(image, self.template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        image = image[int(max_loc[1]):int(max_loc[1]) + self.template.shape[0],
                      int(max_loc[0]):int(max_loc[0]) + self.template.shape[1]]
        image = image[self.crop_center[0]-self.crop_size[0]//2:self.crop_center[0]+self.crop_size[0]//2,
                      self.crop_center[1]-self.crop_size[1]//2:self.crop_center[1]+self.crop_size[1]//2]
        image = self.mask * image
        return image, max_val

def batch_image_bilinear_interpolation(x, coords):
    y_b, y_h, y_w = tf.split(coords, 3, -1)
    y_b, y_h, y_w = tf.reshape(y_b, [-1,1]), tf.reshape(y_h, [-1,1]), tf.reshape(y_w, [-1,1])
    y_h, y_w = tf.clip_by_value(y_h, 0.0, tf.cast(tf.shape(x)[1],"float32") - 1), \
            tf.clip_by_value(y_w, 0.0, tf.cast(tf.shape(x)[2],"float32") - 1)
    
    y_h_r, y_w_r = y_h % 1.0, y_w % 1.0
    y_b, y_h_f, y_w_f = tf.cast(y_b,"int32"), tf.cast(tf.floor(y_h),"int32"), tf.cast(tf.floor(y_w),"int32")
    y_h_f_1, y_w_f_1 = tf.clip_by_value(y_h_f+1, 0, tf.shape(x)[1] - 1),\
                        tf.clip_by_value(y_w_f+1, 0, tf.shape(x)[2] - 1)

    y  = (1 - y_h_r) * (1 - y_w_r) * tf.gather_nd(x, tf.concat([y_b, y_h_f,   y_w_f  ], axis=-1))
    y +=      y_h_r  * (1 - y_w_r) * tf.gather_nd(x, tf.concat([y_b, y_h_f_1, y_w_f  ], axis=-1))
    y += (1 - y_h_r) *      y_w_r  * tf.gather_nd(x, tf.concat([y_b, y_h_f,   y_w_f_1], axis=-1))
    y +=      y_h_r  *      y_w_r  * tf.gather_nd(x, tf.concat([y_b, y_h_f_1, y_w_f_1], axis=-1))
    return tf.reshape(y, tf.shape(x))

def get_batch_image_coords(x):
    coords = tf.meshgrid(tf.range(tf.shape(x)[0]), tf.range(tf.shape(x)[1]), tf.range(tf.shape(x)[2]))
    coords = tf.transpose(coords, (2, 1, 3, 0))
    coords = tf.cast(coords,"float32")
    return coords

def image_warp(x, flow):
    coords = get_batch_image_coords(x) - tf.concat([tf.zeros((tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], 1)), flow], axis=-1)
    y = batch_image_bilinear_interpolation(x, coords)
    return y

def preprocessing(x):
    y = tf.cast(x, "float32") / 127.5 - 1
    return y

def augmentation(x, x_mask = None):
    y = tf.image.random_contrast(x,0.9,1.1)
    y = tf.image.random_brightness(y, 0.2)
    if x_mask is not None:
        y = y * x_mask + x * (1-x_mask)
    return y
    
def get_ng_images(x, x_mask=None):
    log2 = tf.math.log(2.0)
    noise_scale = tf.random.uniform([], 0.0, 1.0)
    batch_size = tf.shape(x)[0]
    x_h_f = tf.cast(tf.shape(x)[1], "float32")
    x_w_f = tf.cast(tf.shape(x)[2], "float32")
    noise_h = tf.cast(2 ** (noise_scale * tf.math.log(x_h_f)/log2), "int32")
    noise_w = tf.cast(2 ** (noise_scale * tf.math.log(x_w_f)/log2), "int32")
    
    mask = tf.random.uniform((batch_size, noise_h, noise_w, 1), -2.0, 2.0)
    mask = tf.image.resize(mask, tf.shape(x)[1:3], tf.image.ResizeMethod.BICUBIC)
    mask = tf.clip_by_value(mask, 0.0, 1.0)

    saturation = tf.random.uniform((batch_size, noise_h, noise_w, 1), 0.0, 1.0)
    noise =    saturation  * tf.random.uniform((batch_size, noise_h, noise_w, 3), -1.0, 1.0) + \
            (1-saturation) * tf.random.uniform((batch_size, noise_h, noise_w, 1), -1.0, 1.0)
    noise = tf.image.resize(noise, tf.shape(x)[1:3], tf.image.ResizeMethod.BICUBIC)
    noise = tf.clip_by_value(noise, -1.0, 1.0)

    flow_h = tf.random.uniform((batch_size, noise_h, noise_w, 1), -x_h_f/tf.cast(noise_h,"float32"), x_h_f/tf.cast(noise_h,"float32"))
    flow_w = tf.random.uniform((batch_size, noise_h, noise_w, 1), -x_w_f/tf.cast(noise_w,"float32"), x_w_f/tf.cast(noise_w,"float32"))
    flow = tf.concat([flow_h,flow_w],axis=-1)
    flow = tf.image.resize(flow, tf.shape(x)[1:3], tf.image.ResizeMethod.BICUBIC)
    
    ng_image = mask * noise + (1-mask) * tf.cast(x, "float32")
    ng_image = image_warp(ng_image, mask*flow)
    ng_image = tf.clip_by_value(ng_image, -1.0, 1.0)
    
    if x_mask is not None:
        ng_image = ng_image * x_mask + x * (1-x_mask)

    return ng_image

def build_model(image_size, filters=16):

    def block(x, filters):
        y = Conv2D(filters, (3, 3), padding="same")(x)
        y = LeakyReLU()(y)
        y = Conv2D(filters, (3, 3), padding="same")(y)
        y = LeakyReLU()(y)
        y = Add()([y, x])
        return y

    def up_block(x, filters):
        y = Conv2D(filters*4, (3, 3), padding="same")(x)
        y = Lambda(tf.nn.depth_to_space,arguments={"block_size":2})(y)
        return y

    def down_block(x, filters):
        y = Lambda(tf.nn.space_to_depth,arguments={"block_size":2})(x)
        y = Conv2D(filters, (3, 3), padding="same")(y)
        return y

    x = Input((image_size[1], image_size[0], 3))
    y = down_block(x,filters)
    f1 = block(y,filters)
    y = down_block(f1,filters*2)
    f2 = block(y,filters*2)
    y = down_block(f2,filters*4)
    f3 = block(y,filters*4)
    y = down_block(f3,filters*8)
    f4 = block(y,filters*8)
    y = down_block(f4,filters*16)
    f5 = block(y,filters*16)

    y = block(f5,filters*16)
    y = Add()([y,f5])
    y = up_block(y,filters*8)
    y = block(y,filters*8)
    y = Add()([y,f4])
    y = up_block(y,filters*4)
    y = block(y,filters*4)
    y = Add()([y,f3])
    y = up_block(y,filters*2)
    y = block(y,filters*2)
    y = Add()([y,f2])
    y = up_block(y,filters)
    y = block(y,filters)
    y = Add()([y,f1])
    y = up_block(y,3)
    y = Add()([y,x])

    return Model(x, y)

def build_train_steps(image_size, model, optimizer, input_mask=None, batch_size=1):
    
    if input_mask is None:
        input_mask = np.ones((1,image_size[1],image_size[0],1),"float32")
    else:
        input_mask = np.reshape(input_mask, (1,image_size[1],image_size[0],1))
    input_mask = tf.Variable(input_mask, trainable=False, dtype="float32")
    
    @tf.function    
    def train_step(ok_images):
        # tf.keras.backend.set_learning_phase(1)
        ok_images = preprocessing(ok_images)
        ok_images = tf.concat([augmentation(i, input_mask) for i in tf.split(ok_images,batch_size,axis=0)], axis=0)
        ng_images = tf.concat([get_ng_images(i, input_mask) for i in tf.split(ok_images,batch_size,axis=0)], axis=0)        
        mask_sum = tf.reduce_sum(tf.tile(input_mask,[tf.shape(ok_images)[0],1,1,tf.shape(ok_images)[3]]))+1

        loss = (tf.reduce_sum(input_mask*tf.abs(model(ng_images)-ok_images))+1) / mask_sum

        adv_gradients = tf.gradients(loss, [ng_images])[0]
        adv_ng_images = ng_images + tf.random.uniform((tf.shape(ng_images)[0], 1, 1, 1), 0.0, 0.001) * tf.sign(adv_gradients)
        adv_ng_images = tf.stop_gradient(adv_ng_images)
        
        adv_loss = (tf.reduce_sum(input_mask*tf.abs(model(adv_ng_images)-ok_images))+1) / mask_sum

        gradients = tf.gradients(adv_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        return loss
    
    @tf.function
    def test_step(ok_images):
        # tf.keras.backend.set_learning_phase(0)
        ok_images = preprocessing(ok_images)
        ok_images = tf.concat([augmentation(i, input_mask) for i in tf.split(ok_images,batch_size,axis=0)], axis=0)
        mask_sum = tf.reduce_sum(tf.tile(input_mask,[tf.shape(ok_images)[0],1,1,tf.shape(ok_images)[3]]))+1
        loss = (tf.reduce_sum(input_mask*tf.abs(model(ok_images)-ok_images))+1) / mask_sum
        return loss
    
    return train_step, test_step

def build_predict_step(image_size, model, input_mask=None):
    
    if input_mask is None:
        input_mask = np.ones((1,image_size[1],image_size[0],1),"float32")
    else:
        input_mask = np.reshape(input_mask, (1,image_size[1],image_size[0],1))
    input_mask = tf.Variable(input_mask, trainable=False, dtype="float32")

    @tf.function
    def predict_step(inputs, threshold, erosion_size):
        x = preprocessing(inputs)
        y = model(x) * input_mask + x * (1-input_mask)
        diff = tf.abs(y-x)/2.0
        diff = tf.reduce_max(diff, -1, True)

        th_diff = tf.cast(diff >= threshold,"float32")
        mask = tf.nn.erosion2d(th_diff, tf.zeros((erosion_size,erosion_size,1), "float32"),strides=(1,1,1,1),dilations=(1,1,1,1),padding="SAME",data_format="NHWC")
        mask = th_diff * tf.nn.dilation2d(mask, tf.zeros((erosion_size+1,erosion_size+1,1), "float32"),strides=(1,1,1,1),dilations=(1,1,1,1),padding="SAME",data_format="NHWC")
        
        diff = tf.tile(diff,[1,1,1,tf.shape(inputs)[3]])
        diff = tf.cast(tf.clip_by_value(diff*255, 0, 255),"uint8")
        mask = tf.tile(mask,[1,1,1,tf.shape(inputs)[3]])
        mask = tf.cast(tf.clip_by_value(mask*255, 0, 255),"uint8")
        x = tf.cast(tf.clip_by_value((x+1)*127.5, 0, 255),"uint8")
        y = tf.cast(tf.clip_by_value((y+1)*127.5, 0, 255),"uint8")
        return x, y, diff, mask
    
    return predict_step

def train(train_image_paths,
          save_weight_path,
          epochs=200,
          batch_size=4):

    data_generator = FitTechA2DataDenerator(train_image_paths, batch_size)
    image_size = (data_generator.mask.shape[1], data_generator.mask.shape[0])
    model = build_model(image_size)
    # model.model.summary()
    train_step, test_step = build_train_steps(image_size,model,Adam(1e-4),data_generator.mask,batch_size)

    epoch_train_loss = []
    epoch_valid_loss = []

    logging.info("Start training")
    progress_bar = tqdm.tqdm(total = epochs * (len(data_generator.train_idx)+len(data_generator.valid_idx)), dynamic_ncols=True, bar_format="Progress:{percentage:3.0f}% [{elapsed}<{remaining}]")    
    for e in range(epochs):
        epoch_train_loss.append([])
        for _ in range(len(data_generator.train_idx)):
            images = data_generator.get_batch_data(True)
            loss = train_step(images)
            epoch_train_loss[-1].append(loss)
            progress_bar.update(1)
        epoch_train_loss[-1] = float(np.mean(epoch_train_loss[-1]))
        
        epoch_valid_loss.append([])
        for _ in range(len(data_generator.valid_idx)):
            images = data_generator.get_batch_data(False)
            loss = test_step(images)
            epoch_valid_loss[-1].append(loss)
            progress_bar.update(1)
        epoch_valid_loss[-1] = float(np.mean(epoch_valid_loss[-1]))
        
        is_model_saved = min(epoch_valid_loss) == epoch_valid_loss[-1]
        if is_model_saved:
            model.save_weights(save_weight_path,save_format="h5")
        
        print("\r",end="")
        logging.info("Checkpoint {}, model_saved:{}, t_l: {:.4f}, v_l: {:.4f}".format(
            e,int(is_model_saved),epoch_train_loss[-1],epoch_valid_loss[-1]))
    
    logging.info("Completed")

def predict(image_paths,
            load_weight_path,
            save_folder,
            threshold,
            erosion_size):
    
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    data_generator = FitTechA2DataDenerator()
    image_size = (data_generator.mask.shape[1], data_generator.mask.shape[0])
    model = build_model(image_size)
    model.load_weights(load_weight_path)

    predict_step = build_predict_step(image_size, model, data_generator.mask)
    
    logging.info("Start testing")
    progress_bar = tqdm.tqdm(image_paths, dynamic_ncols=True, bar_format="Progress:{percentage:3.0f}% [{n_fmt}/{total_fmt}] [{elapsed}<{remaining}]")
    for path in progress_bar:

        image = cv2.imread(path, cv2.IMREAD_COLOR)
        
        if image is None:
            print("\r",end="")
            logging.warning("Failed to read the image. Skip: {}".format(path))
            continue

        image, score = data_generator.template_match(image)
        if score < data_generator.min_match_score:
            print("\r",end="")
            logging.warning("Failed to match the template. Skip: {}".format(path))
            continue
        
        image = np.reshape(image, (1,)+image.shape[:2]+(-1,))
        image, re_image, diff, mask = predict_step(image, threshold, erosion_size)

        save_img = np.concatenate([image[0],re_image[0],diff[0],mask[0]],axis=1)
        if mask[0].numpy().max() > 0:
            save_path = os.path.join(save_folder, "NG_"+os.path.basename(path))
        else:
            save_path = os.path.join(save_folder, "OK_"+os.path.basename(path))
        cv2.imwrite(save_path, save_img)

    logging.info("Completed")