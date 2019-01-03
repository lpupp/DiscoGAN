library(rvest)
setwd("Documents/GitHub/Keras-GAN/discogan/datasets/")

# Harvest tables ==========================================================
base_urls <- c("https://www.architonic.com/en/products/tables/0/3221401", # tables
               "https://www.architonic.com/en/products/seating/0/3221399") # chairs

# tables have 217 pages
# chairs have 400 pages

for (base_url in base_urls){
  cat <- strsplit(base_url, "/")[[1]][6]
  for (i in 1:400){i
    if (cat == 'tables' & i >= 218) break
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
