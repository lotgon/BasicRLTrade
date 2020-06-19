if(!require(devtools)) {install.packages("devtools"); library(devtools)}
if(require(rTTRatesHistory)) {detach("package:rTTRatesHistory", unload=TRUE); remove.packages("rTTRatesHistory")}
install_github("SoftFx/TTWebClient-R",subdir = "rTTRatesHistory")	

install.packages(c('bit64', 'data.table', 'lubridate', 'ggplot2'))
