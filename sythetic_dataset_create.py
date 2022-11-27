from importlib.metadata import files
import random
import shutil
import cv2
from tqdm import tqdm
import cv2, os, sys
import numpy as np
import albumentations as A
from pathlib import Path
import argparse
rand_list = ['True', 'False']

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # az2yolo root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
class album():
    def __init__(self):
        self.class_category = [
        'bad_angle', 
        'blurry', 
        'occlusion', 
        'fogged', 
        'soiled_camera', 
        'dark',
        'bright', 
        'truncated', 
        # 'FLIPPED', 
        'blend',
        'good']
        # self.folder_name = 'oos-test-images-cvs-210\ggo-03005-006-c001' #\oos-test-images-cvs210-non-cvs-coolers' # \ggo-03005-006-c001' #
        self.pose = ['top.jpg', 'middle.jpg', 'bottom.jpg']
        # self.save_dataset_folder = 'class_dataset'

    def change_brightness(self, img_path, value):
        '''
        change the brightness of the image.
        input : image path
        output : store changed image
        '''
        img = cv2.imread(img_path)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v,value)
        v[v > 255] = 255
        v[v < 0] = 0
        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        if value > 0:
            fol = 'bright'
        else:
            fol = 'dark'
        self.store_image(img_path, img, fol)
        # return img

    def visualize(self, image):
        plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.imshow(image)

    def createdataset_folder(self, save_dataset_folder):
        '''
        Delete the previously created dataset folder.
        create folder structure for the images.
        input : dataset path
        '''
        dir_folder = os.getcwd()
        path = os.path.join(dir_folder, save_dataset_folder)
        if os.path.exists(path):
            shutil.rmtree(save_dataset_folder, ignore_errors=True)
        for f in self.class_category:
            os.makedirs(os.path.join(save_dataset_folder,f))

    def store_image(self, img_path, image, class_name):
        '''
        Store the images into argued folder name.
        input : image path, image object, folder name
        output : store the image
        '''
        store_id = img_path.split('\\')[1]
        session_id =  img_path.split('\\')[2]
        pose_name = img_path.split('\\')[3]
        image_save_path = os.path.join(self.save_dataset_folder, class_name) + '\\' + store_id + '_' + session_id + '_' + pose_name + '.jpg'
        # print(image_save_path)
        cv2.imwrite(image_save_path, image)

    def crawl_images(self, folder_name):
        '''
        crawl images in the given session folder.
        input : session folder name
        '''
        for root, dir, files in tqdm(os.walk(folder_name)):
            data = [(os.path.join(root,f)) for f in files]
            if data :
                for img in data:
                    if img.endswith(self.pose[0]):
                        self.preprocess_image(img)
                    if img.endswith(self.pose[1]):
                        self.preprocess_image(img)
                    if img.endswith(self.pose[2]):
                        self.preprocess_image(img)

    def fogged_op(self, image_path):
        '''
        fogged the image.
        input : image path
        output : store fogged image
        '''
        image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = A.RandomSnow (snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=1.5)
        random.seed(7)
        augmented_image = transform(image=image)['image']
        self.store_image(image_path, augmented_image, 'fogged')

    def soiled_camera(self, image_path):
        '''
        bad camera the image.
        input : image path
        output : store bad image
        '''
        image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = A.Compose([A.Spatter (mean=0.35, std=0.3, gauss_sigma=3, cutout_threshold=0.38, intensity=0.3, mode='rain'),
        A.RandomRain (drop_width=1, blur_value=5)])
        random.seed(7)
        augmented_image = transform(image=image)['image']
        self.store_image(image_path, augmented_image, 'soiled_camera')

    def randomcrop(self, image_path):
        '''
        Random Crop Images
        img: image path
        scale: percentage of cropped area
        '''
        scale=0.7
        img = cv2.imread(image_path)
        h, w = int(img.shape[0]*scale), int(img.shape[1]*scale)
        x = random.randint(0, img.shape[1] - int(w))
        y = random.randint(0, img.shape[0] - int(h))
        cropped = img[y:y+h, x:x+w]
        resized = cv2.resize(cropped, (img.shape[1], img.shape[0]))
        self.store_image(image_path, resized, 'truncated')

    def blurry(self, image_path):
        '''
        Add blurry image.
        input : image path
        output : store bad image
        '''
        image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = A.Compose([
            A.Defocus(radius=(5,10)),
            # actual blurry y value is changing resultseg. 15 is very blurry
            # A.GlassBlur (sigma=0.7, max_delta=2, iterations=3),
            # like behind the glass
            # A.MedianBlur (blur_limit=11),
            # nothing happening
            # A.MotionBlur (blur_limit=1001, allow_shifted = True),
            # nothing visible
            # A.ZoomBlur (max_factor=1.31, step_factor=(0.01, 0.03)),
            # zoomed in center and noisy
            # A.OpticalDistortion(distort_limit=0.95, shift_limit=0.75, interpolation=1, border_mode=4),
            # not changing
            # A.PiecewiseAffine(scale=(0.01, 0.015), nb_rows=4, nb_cols=20, interpolation=1, mask_interpolation=1, cval=0, cval_mask=0, mode='constant', absolute_scale=False),
            # too much wavy
            # A.ColorJitter (brightness=0.1, contrast=0.2, saturation=0.2, hue=0.2),
            # very slight change
            # A.PixelDropout (dropout_prob=0.01, per_channel=False, drop_value=100, mask_drop_value=None),
            # noisy image like bad camera images but less
            # A.RandomFog (fog_coef_lower=0.3, fog_coef_upper=1, alpha_coef=0.08),
            # too much blurry nothing is visible
        ])
        random.seed(7)
        augmented_image = transform(image=image)['image']
        self.store_image(image_path, augmented_image, 'blurry')


    def flip_op(self, image_path):
        '''
        flip the image.
        input : image path
        output : store flipped image
        '''
        image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = A.HorizontalFlip(p=0.5)
        random.seed(7)
        augmented_image = transform(image=image)['image']
        self.store_image(image_path, augmented_image, 'flipped')

    def occlusion_op(self, image_path):
        '''
        create occulusion in the image.
        input : image path
        output : store occuled image
        '''
        image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sel = random.choice(rand_list) 
        if sel == 'True':
            transform =  A.RandomSunFlare (flare_roi=(0.3, 0.3, 0.4, 0.5), angle_lower=0.3, angle_upper=0.5, num_flare_circles_lower=1, num_flare_circles_upper=2, src_radius=550, src_color=(100,100,100))
        else:
            transform = A.Superpixels (p_replace=0.1, n_segments=100, max_size=256, interpolation=1)
        random.seed(7)
        augmented_image = transform(image=image)['image']
        self.store_image(image_path, augmented_image, 'occlusion')

    def blend(self, image_path):
        alpha = 0.4
        image = cv2.imread(image_path)
        customer1 = cv2.imread('customer-2.jpg')
        customer1 = cv2.resize(customer1, (1280,960))
        beta = (1.0 - alpha)
        dst = cv2.addWeighted(image, alpha, customer1, beta, 0.0)
        self.store_image(image_path, dst, 'blend')


    def bad_angle(self, image_path):
        '''
        create bad angle perspective transformation in the image.
        input : image path
        output : store occuled image
        '''
        image = cv2.imread(image_path)
        # print(image.shape) # 960, 1280
        
        sel = random.choice(rand_list)
        # left skewed 
        if sel == 'True':
            input_pts = np.float32([[56,65],[368,52],[28,387],[389,390]])
            output_pts = np.float32([[50,50],[350,0],[0,300],[350,350]])
        # right skewed
        else:
            input_pts = np.float32([[156,65],[368,52],[108,387],[389,390]])
            output_pts = np.float32([[50,0],[650,20],[20,550],[650,470]])

        # Compute the perspective transform M
        M = cv2.getPerspectiveTransform(input_pts,output_pts)

        # Apply the perspective transformation to the image
        out = cv2.warpPerspective(image,M,(image.shape[1], image.shape[0]),flags=cv2.INTER_LINEAR)

        # Display the transformed image
        # plt.imshow(out)
        self.store_image(image_path, out, 'bad_angle')

    def original(self, image_path):
        '''
        Store the original image.
        '''
        image = cv2.imread(image_path)
        self.store_image(image_path, image, 'good')

    def preprocess_image(self, img):
        '''
        declare different augmentation function here.
        '''
        self.original(img)
        # flip_op(img)
        self.change_brightness(img, value=140)
        self.change_brightness(img, value=-50)
        self.occlusion_op(img)
        self.fogged_op(img)
        self.soiled_camera(img)
        self.blurry(img)
        self.randomcrop(img)
        self.bad_angle(img)
        self.blend(img)

    def parse_opt(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset-folder', default= 'class_dataset', type=str, help='Path to output datset folder')
        parser.add_argument('--session-images', type=str, help='path to the sessions images')
        parser.add_argument('--pose', type=str, default=['top.jpg', 'middle.jpg', 'bottom.jpg'], help='type of image poses')
        opt = parser.parse_args()
        return opt

    def main(self, opt):
        # dataset folder = store the images in this folder
        self.save_dataset_folder = opt.dataset_folder
        # image path = session path
        if os.path.exists(str(opt.session_images)):
            self.session_folder_name = opt.session_images
        else:
            print("Enter the session images path : --session_images")        
        self.createdataset_folder(self.save_dataset_folder)
        self.crawl_images(self.session_folder_name)

if __name__ == '__main__':
    album = album()
    opt = album.parse_opt()
    album.main(opt)
    
