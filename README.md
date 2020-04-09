# BasicRLTrade
Test Q-Value and Actor-Critics algorithms in Forex domain.  
- Quotes Source - QuotesDownloader Github project.
- There are custom embedding price sequence based on k-means
- Stochastic is added to the environment

## Prerequsites:
- Windows OS, R
- run prerequisite.R to install all dependacies. Non cran package will be installed too [rTTRatesHistory](https://github.com/SoftFx/TTWebClient-R)


## Project Structure:
- ~\tools\QuotesDownloader\ tools from [QuotesDownloader](https://github.com/SoftFx/QuotesDownloader/releases)
- ~\QuotesData\ quotes file cache


## First Steps
1. Embedding quote series. First step is building k-cluster means. Run function BuildCenterLines(c("EURUSD", "EURCHF"), startDate=ISOdate(2017, 6, 1, tz = "UTC"), endDate=ISOdate(2019, 6, 1, tz = "UTC"), "M15", cluster_number=50, 360, outputName="cTable1200_EuroSymbols"). Result will be saved to saveRDS(.., "cTable1200_EuroSymbols") file. 
Cluster Number is embedding size.
Test the result with command: ggplot(cTable, aes(x=index, y=value, group=clusterIndex, colour=clusterIndex, size=clusterPower)) + geom_line().
2. 

