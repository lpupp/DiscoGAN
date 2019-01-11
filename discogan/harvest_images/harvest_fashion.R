library(rvest)
setwd("/Users/lucagaegauf/Documents/GitHub/Keras-GAN/discogan/datasets/")

# Harvest belts ===========================================================
# base_urls <- c("https://www.bing.com/images/search?q=belts+fabric&go=Search&qs=n&form=QBIR&qft=+filterui%3Aimagesize-custom_64_64+filterui%3Aphoto-photo+filterui%3Aaspect-square+filterui%3Acolor2-FGcls_WHITE&sp=-1&pq=belts+fabric&sc=0-12&sk=&cvid=0A54B2AEA5744EC1BFD7E3AB2927C0FF", # fabric
#                "https://www.bing.com/images/search?q=belts+kids&go=Search&qs=n&form=QBIR&qft=+filterui%3Aimagesize-custom_64_64+filterui%3Aphoto-photo+filterui%3Aaspect-square+filterui%3Acolor2-FGcls_WHITE&sp=-1&pq=belts+childen&sc=1-13&sk=&cvid=8F3E20CEC7FD4367B0C58DCF0511CB7C", # kids
#                "https://www.bing.com/images/search?q=belts+mens&go=Search&qs=n&form=QBIR&qft=+filterui%3Aimagesize-custom_64_64+filterui%3Aphoto-photo+filterui%3Aaspect-square+filterui%3Acolor2-FGcls_WHITE&sp=-1&pq=belts+men&sc=0-9&sk=&cvid=7393DA3AFCE64A6BA22DCD63007E7753", # men
#                "https://www.bing.com/images/search?q=belts+womens&go=Search&qs=n&form=QBIR&qft=+filterui%3Aimagesize-custom_64_64+filterui%3Aphoto-photo+filterui%3Aaspect-square+filterui%3Acolor2-FGcls_WHITE&sp=-1&pq=belts+womens&sc=0-12&sk=&cvid=0A7CCF84972F40B7B94C57E068D7070D", # women
#                "https://www.bing.com/images/search?q=belts+fashion&go=Search&qs=n&form=QBIR&qft=+filterui%3Aimagesize-custom_64_64+filterui%3Aphoto-photo+filterui%3Aaspect-square+filterui%3Acolor2-FGcls_WHITE&sp=-1&pq=belts+fashi&sc=1-11&sk=&cvid=64464A40E7A3413DBA4F39A360713706", # fashion
#                "https://www.bing.com/images/search?q=formal+belt&go=Search&qs=n&form=QBIR&qft=+filterui%3Aimagesize-custom_64_64+filterui%3Aphoto-photo+filterui%3Aaspect-square+filterui%3Acolor2-FGcls_WHITE&sp=-1&pq=formal+belt&sc=0-11&sk=&cvid=C916D88056E3447386ABAACD037AD013") # formal
# 
# cats <- c("fabric", "kids", "mens", "womens", "fashion", "formal")
# rows <- c(103, 88, 86, 86, 97, 48)
# #if_not_square(padd)
# #resize
# 
# for (cat in 1:length(base_urls)){
#   base_url <- base_urls[cat]
#   overview <- read_html(base_url)
#   for (row in 1:rows[cat]){
#     for(col in 1:12){
#       
#     }
#   }
#   xx <- overview %>%
#     html_nodes("body") %>% 
#     html_nodes(xpath='//*[@id="b_content"]') %>%
#     html_nodes("canvas")
#   list_of_tables <- html_nodes(overview, '//*[@id="mmComponent_images_1"]/ul[2]/li[4]/div/div/a/div/img')
#   #list_of_table_urls <- html_attr(html_children(list_of_tables), "href")
#   list_of_tableimg_urls <- html_attr(html_nodes(html_nodes(list_of_tables, "a"), "img"), "src")
#   list_of_tablejpg_urls <- list_of_tableimg_urls[!grepl(".gif", list_of_tableimg_urls)]
#   
#   for (url in list_of_tablejpg_urls){
#     img_name <- tail(strsplit(url, "/")[[1]], 1)
#     img_name <- gsub("-", "_", img_name)
#     save_img <- try(download.file(url, destfile = paste0("furniture/", cat, "/", img_name)))
#   }
# }

# Shopstyle belts =========================================================
require(RSelenium)
library(stringr)
library(imager)

base_urls <- c("https://www.shopstyle.co.uk/browse/belts",
               "https://www.shopstyle.co.uk/browse/mens-belts")
cats <- c("f", "m")

df_output <- read.csv("fashion/harvested_belts.csv", stringsAsFactors=FALSE)
df_output$X<- NULL

img_target <- function(x){ #, i, cat, n=4){
  filename <- strsplit(x, "/")
  filename <- filename[[1]][length(filename[[1]])]
  #img_type <- substr(x, nchar(x)-n+1, nchar(x))
  #paste0("fashion/belts/", cat, str_pad(i, 6, pad="0"), img_type)
  paste0("fashion/belts/", filename)
}

j <- 2
cat <- cats[j]
base_url <- base_urls[j]

rD <- rsDriver(browser="firefox")
remDr <- rD[["client"]]
remDr$navigate(base_url)

v_url <- c()
#v_url <- df_output$urltxt 

for(i in 1:10000){
  xpath <- paste0("/html/body/app-root/div[1]/ss-search-page/div/div[2]/div[2]/ss-products-list/div/span[", i , "]/ss-product-cell/div/div/a[3]/img")
  tmp <- try(remDr$findElement(using="xpath", value=xpath))
  if(class(tmp) != "try-error"){
    img_url <- tmp$getElementAttribute('src')[[1]]
    if (! img_url %in% v_url){
      
      img <- load.image(img_url)# %>% plot
      #img %>% plot
      
      w <- width(img)
      h <- height(img)
      
      if (h > w) {
        n <- h/256
        img <- resize(img, size_x=as.integer(w/n), size_y=256, interpolation_type=3) #%>% plot
      } else if (w > h) {
        n <- w/256
        img <- resize(img, size_x=256, size_y=as.integer(h/n), interpolation_type=3) #%>% plot
      } 
      
      # Save image
      img_tar <- img_target(img_url)
      save.image(img, img_tar)
    }
    v_url <- c(v_url, img_url)
  }
}

#df_output <- data.frame(urltxt = unique(unlist(v_url)), harvested=0)
#write.csv(df_output, "fashion/harvested_belts.csv")

remDr$close()

