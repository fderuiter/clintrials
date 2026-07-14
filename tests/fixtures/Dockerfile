FROM r-base:4.3.2

RUN R -e "install.packages(c('remotes', 'mvtnorm'), repos='http://cran.rstudio.com/'); remotes::install_version('bcrm', version='0.5.4', repos='http://cran.rstudio.com/')"
