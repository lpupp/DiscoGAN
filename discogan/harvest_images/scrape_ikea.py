# TODO
from selenium import webdriver
import time
from urllib.request import urlretrieve
import os
from itertools import compress
import json
import numpy as np


def _safe_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

save_path = '/Users/lucagaegauf/Dropbox/GAN/fashion/zalando/shoes/'
links_path = os.path.join(save_path, 'output_links')
_safe_mkdir(save_path)
_safe_mkdir(links_path)

# TODO (lpupp) get all shoes
shoes = ['ankle-boots'
#         'Ballet Pumps'
#         'Boots'
#         'Flats & Lace-Ups'
#         'Flip Flops & Beach Shoes'
#         'Heels
#         'Mules & Clogs
#         'Outdoor Shoes
#         'Sandals
#         'Shoe Care
#         'Slippers
#         'Sports Shoes
#         'Trainers
         ]

#my_list =['?shop=']

for shoe in shoes:
    page_num = 1
    shoe_out = {}
    while True:
        driver = webdriver.Firefox(executable_path='/usr/local/Cellar/geckodriver/0.23.0/bin/geckodriver')
        url = 'https://www.zalando.co.uk/womens-shoes-' + shoe + '/?p=' + str(page_num)
        driver.get(url)
        print('Fetching URL: ' + url)

        if page_num > 1:
            true_page_num = int(driver.current_url.split('/?p=')[1])
            if true_page_num < page_num:
                print('Final page reached')
                break

        print('Page Loading Complete for URL: ' + url)

        time.sleep(60)
        products = driver.find_elements_by_class_name('cat_imageLink-OPGGa')
        hrefs = [prod.get_attribute('href') for prod in products]
        print('Total ' + str(len(hrefs)) + ' images on page ' + str(page_num))

        def get_img(url):
            driver.get(url)
            images = driver.find_elements_by_tag_name('img')
            img_ids = [img.get_attribute('id') for img in images]
        
            images = list(compress(images, ['galleryImage' in i for i in img_ids]))
            img_src = [img.get_attribute('src') for img in images]
            img_src = list(compress(img_src, ['packshot' in i for i in img_src]))[0]
            return img_src
            
        # Drop preffix 'https://www.zalando.co.uk/'
        shoe_out.update(dict((k.split('co.uk/')[1], get_img(k)) for k in hrefs))
        filename_links = str(shoe + '.json')
        full_filename_links = os.path.join(links_path, filename_links)

        with open(full_filename_links, 'w') as fp:
            json.dump(shoe_out, fp)

        print('File Writing Complete for ' + shoe + ' page ' + str(page_num))

        driver.close()

# -----------------------------------------------------------------------------------------------

all_files = os.listdir(links_path)
    
shoes = [e for e in all_files if '.DS' not in e]

for shoe in shoes:
    file = [e for e in all_files if shoe in e]
    file_path = os.path.join(links_path, file)
        
    with open(file_path, 'r') as fp:
        shoe_dict = json.load(fp)

    print('There are {} shoes in {} category'.format(len(shoe_dict), shoe))

    save_sub_path = os.path.join(save_path, shoe)
    _safe_mkdir(save_sub_path)
    
    i = 0
    for image_loc in shoe_dict.values():
        filename = str(i) + '.jpg'
        full_filename = os.path.join(save_sub_path, filename)
        try:
            urlretrieve(url=image_loc, filename=full_filename)
        except Exception as e:
            print('ERROR: ' + str(i))
            continue
        i += 1
        print('image' + str(i))

# -----------------------------------------------------------------------------------------------
# Get recommendations
save_path = '/Users/lucagaegauf/Dropbox/GAN/fashion/zalando/shoes/'
links_path = os.path.join(save_path, 'output_links')
_safe_mkdir(save_path)
_safe_mkdir(links_path)

all_files = os.listdir(links_path)
    
shoes = [e for e in all_files if '.DS' not in e]

for shoe in shoes:
    file = [e for e in all_files if shoe in e]
    file_path = os.path.join(links_path, file)
        
    with open(file_path, 'r') as fp:
        shoe_dict = json.load(fp)

    save_sub_path = os.path.join(save_path, shoe)
    _safe_mkdir(save_sub_path)

    recommend_dict = {}

    i = 0
    for k in shoe_dict:

        driver = webdriver.Firefox(executable_path='/usr/local/Cellar/geckodriver/0.23.0/bin/geckodriver')
        url = 'https://www.zalando.co.uk/' + k
        driver.get(url)
        print('Fetching URL: ' + url)

        time.sleep(60)
        images = driver.find_elements_by_tag_name('img')
        img_desc = [img.get_attribute('alt') for img in images]
        
        # TODO(lpupp) improve
        bag = np.array(['bag' in desc for desc in img_desc])
        dress = np.array(['dress' in desc for desc in img_desc])
        belt = np.array(['belt' in desc for desc in img_desc])
        
        recommended = list(compress(images, bag | dress | belt))

        recommend_dict[k] = [e.get_attribute('src') for e in recommended]
        
        if i % 100 = 0:
            filename_links = str(shoe + '.json')
            full_filename_links = os.path.join(links_path, "recommendations", filename_links)

            with open(full_filename_links, 'w') as fp:
                json.dump(recommend_dict, fp)

            print('File Writing Complete for ' + shoe + ' iteration ' + str(i))

        driver.close()
