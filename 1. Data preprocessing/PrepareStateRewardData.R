source("BuildClusters.R")
require(ggplot2)

#Generate kmeans lines for time series embedding
#BuildCenterLines(c(tradedSymbol), embeddingStartDate, embeddingEndDate, periodicity, cluster_number, window_size, outputName=mainCTableFileName)
#review result 
#meltedcTable <- readRDS(meltedCTable_FileName)
#ggplot(meltedcTable[symbol==tradedSymbol & clusterIndex ==11], aes(x=index, y=value, group=clusterIndex, colour=clusterIndex, size=clusterPower)) + geom_line()


BuildAgentData <- function(quotes, window_size, inputRawCenters_FileName, outputAgentData_FileName){
  rawCenters <- fread(file=inputRawCenters_FileName)
  embeddingSize <-rawCenters[,.N]
  bids <- quotes[,.(price = close.bid)]
  startIndices <- seq.int(1, nrow(bids) - window_size, by=1)
  m <- matrix(0, nrow = length(startIndices), ncol = embeddingSize)
  for(i in seq_along(startIndices)){
    meanWindow <- bids[startIndices[i]:(startIndices[i]+window_size-1), mean(price)]
    norm_price <- bids[startIndices[i]:(startIndices[i]+window_size-1), price/meanWindow]
    m[i,] <- rowSums(sweep(rawCenters, 2, norm_price)^2)
  }
  assertthat::assert_that(quotes[startIndices + window_size, .N]== nrow(m))
  embeddedQuotes <- cbind(quotes[startIndices + window_size,], m)
  fwrite(embeddedQuotes, outputAgentData_FileName)
  embeddedQuotes
}

BuildEnvironmentData <- function(startDate=ISOdate(2017, 6, 1, tz = "UTC"), endDate=ISOdate(2020, 1, 1, tz = "UTC"),
                                 symbol, periodicity, outputEnvData_FileName){
  quotes <- GetQuotes(symbol, startDate, endDate, periodicity, "BidAsk")
  quotes <- quotes[, .(datetime, close.ask, close.bid, high.ask, low.bid)]
  fwrite(quotes, outputEnvData_FileName)
  quotes
}
Convert2Simulation <- function(EnvData_FileName, simulationType = "sin", window_size){
  quotes <- fread(EnvData_FileName)
  if( simulationType == "sin"){
    med <- quotes[1:(7*window_size), median(close.ask)]
    max <- quotes[1:(7*window_size), max(close.ask)]
    quotes[,newclose.ask:=sin(.I*pi/2/window_size) * (max-med) + med]
    quotes[,close.bid:= newclose.ask - (close.ask-close.bid)]
    quotes[,high.ask:= newclose.ask - (close.ask-high.ask)]
    quotes[,low.bid := newclose.ask - (close.ask-low.bid)]
    quotes[,close.ask:=newclose.ask]
    quotes[,newclose.ask:=NULL]
  }
  if( simulationType == "mirror"){
    #we replicate 200 times to avoid bias at the end of period. Bot could create huge position without consequences.
    smallIndex <- rep(c(1:(3*window_size), (3*window_size):1), 200)
    quotes <- quotes[smallIndex,]
  }
  fwrite(quotes, EnvData_FileName)
  quotes
}



