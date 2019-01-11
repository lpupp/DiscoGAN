library(rvest)
setwd("/Users/lucagaegauf/Documents/GitHub/Keras-GAN/discogan/datasets/")

# Harvest tables ==========================================================
base_urls <- c("https://www.architonic.com/en/products/tables/0/3221401", # tables
               "https://www.architonic.com/en/products/seating/0/3221399", # chairs
               "https://www.architonic.com/en/products/storage/0/3221372", # storage
               "https://www.architonic.com/en/products/carpets-rugs/0/3220753") # carpets

pages <- list("tables" = 217, "seating" = 400, "storage" = 121, "carpets-rugs" = 144)

for (base_url in base_urls[3:4]){
  cat <- strsplit(base_url, "/")[[1]][6]
  for (i in 1:400){i
    if (i >= pages[[cat]] + 1) break
    overview <- read_html(paste0(base_url, "/", i))
    list_of_tables <- html_nodes(overview, ".product-overview-container")
    #list_of_table_urls <- html_attr(html_children(list_of_tables), "href")
    list_of_tableimg_urls <- html_attr(html_nodes(html_nodes(list_of_tables, "a"), "img"), "src")
    list_of_tablejpg_urls <- list_of_tableimg_urls[!grepl(".gif", list_of_tableimg_urls)]
    
    for (url in list_of_tablejpg_urls){
      img_name <- tail(strsplit(url, "/")[[1]], 1)
      img_name <- gsub("-", "_", img_name)
      save_img <- try(download.file(url, destfile = paste0("furniture/", cat, "/", img_name)))
    }
  }
}
