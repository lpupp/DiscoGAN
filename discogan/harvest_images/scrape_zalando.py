from selenium import webdriver
import time
from urllib.request import urlretrieve
import os
from itertools import compress
import json
#import numpy as np

def _safe_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


save_path = '/Users/lucagaegauf/Dropbox/GAN/fashion/zalando/'
_safe_mkdir(save_path)
    
products = {'shoes': [#'womens-shoes-ankle-boots',
                      #'womens-shoes-ballet-pumps',
                      #'womens-shoes-boots',
                      #'womens-shoes-flats-lace-ups',
                      #'flip-flops-beach-shoes',
                      #'womens-shoes-heels',
                      #'womens-shoes-mules-clogs',
                      #'womens-outdoor-shoes',
                      'womens-shoes-sandals',
                      'womens-shoes-slippers',
                      'womens-sports-shoes',
                      'womens-shoes-trainers'],
    #'belts': ['bags-accessories-womens-belts',
    #          'premium-womens-belts'],
    'bags': [#'clutch-bags',
             #'handbags',
             #'shopping-bags',
             #'shoulder-bags',
             #'rucksacks',
             'sports-travel-bags',
             'womens-sports-bags',
             'womens-drawstring-bags',
             'womens-shoulder-bags',
             'laptop-bags',
             'wash-bags',
             'womens-luggage',
             'womens-daysacks',
             'womens-touring-rucksacks',
             'womens-trekking-rucksacks',
             'premium-womens-backpacks',
             'premium-womens-clutches',
             'premium-womens-handbags',
             'premium-womens-make-up-bags',
             'premium-womens-shoulder-bags',
             'premium-womens-sports-bags',
             'premium-womens-shopping-bags',
             'premium-womens-travel-bags',
             'premium-womens-purses',
             'purses'],
    'dresses': ['summer-dresses',
                'cocktail-dresses',
                'denim-dresses',
                'jersey-dresses',
                'knitted-dresses',
                'maxi-dresses',
                'shirt-dresses',
                'work-dresses',
                'premium-womens-dresses']
    }

for k in products:
    print(k)
    save_k_path = os.path.join(save_path, k)
    links_path = os.path.join(save_k_path, 'output_links')
    _safe_mkdir(save_k_path)
    _safe_mkdir(links_path)

#%%
page_start = 1
max_page = 200

for cat, sub_cats in products.items():
    links_path = os.path.join(save_path, cat, 'output_links')
    
    for sub_cat in sub_cats: 
        start = time.time()
        page_num = page_start
        shoe_out = {}
        while True:
            #shoe_out = {}
            driver = webdriver.Firefox(executable_path='/usr/local/Cellar/geckodriver/0.23.0/bin/geckodriver')
            url = 'https://www.zalando.co.uk/' + sub_cat + '/?p=' + str(page_num)
            driver.get(url)
            print('Fetching URL: ' + url)
    
            if page_num > 1:
                true_page_num = int(driver.current_url.split('/?p=')[1])
                if true_page_num < page_num:
                    print('Final page reached')
                    break
            if page_num == max_page:
                break
    
            print('Page Loading Complete for URL: ' + url)
    
            time.sleep(10)
            prods = driver.find_elements_by_class_name('cat_imageLink-OPGGa')
            hrefs = [prod.get_attribute('href') for prod in prods]
            hrefs = hrefs[:10]
            print('Total ' + str(len(hrefs)) + ' images on page ' + str(page_num))
    
            def get_img(url):
                driver.get(url)
                time.sleep(3)
                buttons = driver.find_elements_by_tag_name('button')
                buttons = [bu for bu in buttons if bu.get_attribute('class')=='h-action h-card-media default']
                img_src = []

                i = 0
                while len(img_src) == 0:
                    if i != 0:
                        buttons[i].click()
                        time.sleep(2)
                    images = driver.find_elements_by_tag_name('img')
                    imgs_src = [img.get_attribute('src') for img in images]
                    imgs_src = list(filter(lambda e: e is not None, imgs_src))
                    img_src = [img for img in imgs_src if 'packshot/pdp-gallery' in img or 'packshot/pdp-zoom' in img]
                    print(len(img_src))
                    i += 1
                return img_src[0]
                    
            # Drop preffix 'https://www.zalando.co.uk/'
            #shoe_out = dict((k.split('co.uk/')[1], get_img(k)) for k in hrefs)
            shoe_out = {}
            for k in hrefs:
                try:
                    shoe_out[k.split('co.uk/')[1]] = get_img(k)
                except IndexError:
                    continue

            filename_links = str(sub_cat + '_' + str(page_num) + '.json')
            full_filename_links = os.path.join(links_path, filename_links)
    
            with open(full_filename_links, 'w') as fp:
                json.dump(shoe_out, fp)
    
            print('File Writing Complete for ' + sub_cat + ' page ' + str(page_num))
    
            driver.close()
            page_num += 1

        print(time.time() - start)

#%%  
for cat, sub_cats in products.items():
    links_path = os.path.join(save_path, cat, 'output_links')
    all_files = os.listdir(links_path)
    all_files = [e for e in all_files if '.DS' not in e]
    
    for sub_cat in sub_cats:
        file = [e for e in all_files if sub_cat in e]
        file_path = os.path.join(links_path, file[0])
            
        with open(file_path, 'r') as fp:
            cat_dict = json.load(fp)
    
        print('There are {} items in {} category'.format(len(cat_dict), sub_cat))
    
        save_sub_path = os.path.join(save_path, cat, sub_cat)
        _safe_mkdir(save_sub_path)
        
        i = 0
        for image_loc in cat_dict.values():
            filename = str(i) + '.jpg'
            full_filename = os.path.join(save_sub_path, filename)
            try:
                urlretrieve(url=image_loc, filename=full_filename)
            except Exception as e:
                print('ERROR: ' + str(i))
                continue
            i += 1
            print('image' + str(i))

#%%
# -----------------------------------------------------------------------------------------------
## Get recommendations
#bag_cats = ['bag',
#            'clutch-bag',
#            'clutches',
#            'designer-bag',
#            'handbag',
#            'backpacks',
#            'tote-bag',
#            'shoulder-bag',
#            'rucksacks',
#            'travel-bag',
#            'sport-bag',
#            'laptop-bag',
#            'phone-bag',
#            'wash-bag',
#            'make-up-bag',
#            'wallet',
#            'purse',
#            'bumbag',
#            'bum-bag',
#            'across-body-bag',]
#
#dress_cats = ['dress',
#              'casual-dress',
#              'cocktail-dress',
#              'denim-dress',
#              'jersey-dress',
#              'knitted-dress',
#              'maxi-dress'
#              'shirt-dress'
#              'work-dress']
#belt_cats = ['belt']
#
#save_path = '/Users/lucagaegauf/Dropbox/GAN/fashion/zalando/shoes/'
#links_path = os.path.join(save_path, 'output_links')
#rec_path = os.path.join(save_path, 'recommendations')
#_safe_mkdir(save_path)
#_safe_mkdir(links_path)
#_safe_mkdir(rec_path)
#
#all_files = os.listdir(links_path)
#all_files = [e for e in all_files if '.DS' not in e]
#
#for shoe in shoes:
#    file = [e for e in all_files if shoe in e]
#    file_path = os.path.join(links_path, file[0])
#        
#    with open(file_path, 'r') as fp:
#        shoe_dict = json.load(fp)
#
#    #save_sub_path = os.path.join(rec_path, shoe)
#    #_safe_mkdir(save_sub_path)
#
#    recommend_dict = {}
#
#    i = 0
#    for k in shoe_dict:
#
#        driver = webdriver.Firefox(executable_path='/usr/local/Cellar/geckodriver/0.23.0/bin/geckodriver')
#        url = 'https://www.zalando.co.uk/' + k
#        driver.get(url)
#        print('Fetching URL: ' + url)
#
#        time.sleep(60)
#        perfect_pairings = driver.find_element_by_xpath('/html/body/div[4]/div/div/div/div[2]/div/div[1]/div/div/div/div[2]/a')
#        driver.get(perfect_pairings.get_attribute('href'))
#        time.sleep(60)
#
#        #images = driver.find_elements_by_tag_name('img')
#        a = driver.find_elements_by_tag_name('a')
#        a_href = [i.get_attribute('href') for i in a]
#        a_href = list(filter(None, a_href))
#        a_href = list(filter(lambda k: len(k.split('/')) == 4, a_href))
#        a_href = list(filter(lambda k: 'https://www.zalando.co.uk/' in k, a_href))
#        
#        recs = []
#        for bag in bag_cats:
#            recs += [ah for ah in a_href if bag in ah]
#        for dress in dress_cats:
#            recs += [ah for ah in a_href if dress in ah]
#        for belt in belt_cats:
#            recs += [ah for ah in a_href if belt in ah]
#        
#        recommend_dict[k] = list(set(recs))
#        
#        if i % 100 = 0:
#            filename_links = str(shoe + '.json')
#            full_filename_links = os.path.join(rec_path, filename_links)
#
#            with open(full_filename_links, 'w') as fp:
#                json.dump(recommend_dict, fp)
#
#            print('File Writing Complete for ' + shoe + ' iteration ' + str(i))
#
#        driver.close()
#
##%%
#all_files = os.listdir(rec_path)
#all_files = [e for e in all_files if '.DS' not in e]
#
#for file in all_files:
#    file_path = os.path.join(rec_path, file)
#        
#    with open(file_path, 'r') as fp:
#        shoe_dict = json.load(fp)
#    
#    for k in shoe_dict:
#        # TODO(lpupp) how should I save this???? it somehow needs to match to a shoe.
#        save_sub_path = os.path.join(rec_path, k)
#        _safe_mkdir(save_sub_path)
#    
#        i = 0
#        for image_loc in shoe_dict.values():
#            filename = str(i) + '.jpg'
#            full_filename = os.path.join(save_sub_path, filename)
#            try:
#                urlretrieve(url=image_loc, filename=full_filename)
#            except Exception as e:
#                print('ERROR: ' + str(i))
#                continue
#            i += 1
#            print('image' + str(i))