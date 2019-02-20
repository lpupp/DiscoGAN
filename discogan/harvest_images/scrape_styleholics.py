from selenium import webdriver
import time
from urllib.request import urlretrieve
import os

def _safe_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

save_path = '/Users/lucagaegauf/Dropbox/GAN/fashion/stylaholic/dresses/'
links_path = os.path.join(save_path, 'output_links')
_safe_mkdir(save_path)
_safe_mkdir(links_path)

shops = [#'yoox',
         #'yoins',
         'about-you',
         'bonprix',
         'cecil',
         'conleys',
         'emp',
         'fashion-id',
         'impressionen',
         'mango',
         'mytheresa',
         'only',
         'sportscheck',
         'street-one',
         'vero-moda'#,
         #'atelier-goldener-schnitt',
         #'bodenmode',
         #'cecil',
         #'galeria-kaufhof',
         #'happy-size',
         #'joop',
         #'street-one'
         ]

dresses = ['abendkleider',
           'brautkleider',
           'cocktailkleider',
           'freizeitkleider',
           'midikleider',
           'maxikleider',
           'minikleider',
           'jeanskleider',
           'businesskleider',
           'sommerkleider',
           'kleider-etuikleider',
           'kleider-jerseykleider',
           'kleider-partykleider',
           'kleider-blusenkleider',
           'kleider-spitzenkleider',
           'strandkleider',
           'trachten-dirndl'
           ]

my_list =['?shop=']

for shop in shops:
    for dress in dresses:
        for element in my_list:
            driver = webdriver.Firefox(executable_path='/usr/local/Cellar/geckodriver/0.23.0/bin/geckodriver')
            # url = 'https://www.stylaholic.de/damenbekleidung-kleider?shop=' + element
            url = 'https://www.stylaholic.de/damenbekleidung-' + dress + element + shop
            driver.get(url)
            print('Fetching URL: ' + url)

            j = 1
            while True:
                try:
                    print('Screen ' + str(j))
                    loadMoreButton = driver.find_element_by_xpath('/html/body/div[3]/div[2]/div[1]/sh-grid/sh-grid-item[2]/div/sh-grid[2]/button')
                    time.sleep(3)
                    loadMoreButton.click()
                    time.sleep(3)
                    print(j)
                    j = j + 1

                except Exception as e:
                    print(e)
                    break

            print('Page Loading Complete for URL: ' + url)

            time.sleep(60)
            images = driver.find_elements_by_tag_name('img')
            print('Total ' + str(len(images)) + ' images')

            # Write to file
            filename_links = str(dress + element + shop + '.txt').replace('?', '')
            full_filename_links = os.path.join(links_path, filename_links)

            text_file = open(full_filename_links, 'a+')

            #text_file.write(element + 'Total' + str(len(images)) + 'images' + '\n')

            for image in images:
                text_file.write(str(image.get_attribute('data-original')) + ' \n')

            text_file.close()

            print('File Writing Complete for URL: ' + element)

            driver.close()

# -----------------------------------------------------------------------------------------------
all_files = os.listdir(links_path)

dresses = list(set([e.split('shop=')[0] for e in all_files if '.DS' not in e]))

array = []
for dress in dresses:
    _files = [e for e in all_files if dress in e]
    for file in _files:
        file_path = os.path.join(links_path, file)
        
        if os.stat(file_path).st_size == 0:
            os.remove(file_path) 
            continue
        
        with open(file_path, 'rt') as fd:
            # TODO(lpupp) if file is empty, delete.
            for line in fd:
                array.append(line)

    print('There are {} dresses in {} category'.format(len(array), dress))

    save_sub_path = os.path.join(save_path, dress)
    _safe_mkdir(save_sub_path)
    
    i = 0
    for image in array:
        filename = str(i) + '.jpg'
        full_filename = os.path.join(save_sub_path, filename)
        image_loc = image.replace('\n', '')
        print(image_loc)
        try:
            urlretrieve(url=image_loc, filename=full_filename)
        except Exception as e:
            print('ERROR: ' + str(i))
            continue
        i += 1
        print('image' + str(i))
