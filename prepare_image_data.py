import os
from PIL import Image



def resize_images(base_dir, processed_images_dir):
    #Define a function called resize_images which loads each image, and resizes it to the same height and width before saving the new version in your new processed_images folder.
    #Set the height of the smallest image as the height for all of the other images. Your code should maintain the aspect ratio of the image and adjust the width proportionally to the change in height, rather than just squashing it vertically. This function should also ensure that every image is in RGB format. If it isn't, discard it.
    image_file_path_list = []
    smallest_height = float('inf')
    
    for sub_dir in os.listdir(base_dir):
        sub_dir_file_path = os.path.join(base_dir, sub_dir)
        if os.path.isdir(sub_dir_file_path):
            for file in os.listdir(sub_dir_file_path):
                if file.endswith('.jpg') or file.endswith('.png'):
                    with Image.open(os.path.join(sub_dir_file_path, file)) as img:
                        if img.mode == 'RGB':
                            image_file_path_list.append(os.path.join(sub_dir_file_path, file))
                            width, height = img.size
                            if height < smallest_height:
                                smallest_height = height
    print(smallest_height)
                        

    for file in image_file_path_list:
        with Image.open(file) as img:
            width, height = img.size
            aspect_ratio = width / height
            new_width = int(aspect_ratio * smallest_height) 
            resized_img = img.resize((new_width, smallest_height))
            resized_img.save(os.path.join(processed_images_dir, os.path.basename(file)))

if __name__ == '__main__':
    if not os.path.exists('./airbnb-property-listings/processed_images'):
        os.mkdir('./airbnb-property-listings/processed_images')

    base_dir = './airbnb-property-listings/images/'
    processed_images_dir = './airbnb-property-listings/processed_images/'
    resize_images(base_dir, processed_images_dir)
    