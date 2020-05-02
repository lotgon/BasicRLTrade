source("DownloadQuotes.R")

#BuildCenterLines(c("EURCHF"), startDate=ISOdate(2017, 6, 1, tz = "UTC"), endDate=ISOdate(2019, 6, 1, tz = "UTC"), "M15", cluster_number=50, 360, outputName="cTable1200_EuroSymbols")
BuildCenterLines <- function(symbols, startDate, endDate, periodicity="M15", cluster_number=50, window_size=1200, 
                             outputRawCenters_FileName, outputMeltedCTable_FileName){
  #for(curr_symbol in symbols){
    
    curr_symbol <- Reduce(paste, symbols)
    m <- NULL
    
    for(j in seq_along(symbols)){
      quotes <- GetQuotes(symbols[[j]], startDate, endDate, periodicity, "Bids")[,.(open)]
      startIndices <- seq.int(1, nrow(quotes) - window_size+1, by=10)
      m_part <- matrix(0, nrow = length(startIndices), ncol = window_size)
      for(i in seq_along(startIndices)){
        meanWindow <- quotes[startIndices[i]:(startIndices[i]+window_size-1), mean(open)]
        m_part[i,] <- quotes[startIndices[i]:(startIndices[i]+window_size-1), open/meanWindow] 
      }
      m <- rbind(m_part, m)
    }
    r <- kmeans(m, cluster_number, iter.max = 40)
    
    centers <- as.data.table(r$centers)
    fwrite(centers, outputRawCenters_FileName)
    centers [,clusterIndex:= seq_len(.N)]
    centers[,clusterPower:=r$size/sum(r$size)]

    
    cTable <- melt(centers, id.vars = c("clusterIndex", "clusterPower"))
    cTable[,index := as.numeric(variable) ]
    cTable[,variable:=NULL]

    cTable[,window_size:=window_size]
    cTable[,cluster_number:=cluster_number]
    cTable[,symbol:=curr_symbol]
    cTable[,startDate:=startDate]
    cTable[,endDate:=endDate]
    
    diskTable <- data.table()
    if(file.exists(outputMeltedCTable_FileName)){
      diskTable <- readRDS(outputMeltedCTable_FileName)
      ws <- window_size
      diskTable <- diskTable[!(symbol==curr_symbol&window_size==ws),]
    }
    diskTable <- rbind(diskTable, cTable)
    
    saveRDS(diskTable, outputMeltedCTable_FileName)
  #}
}

#export data for mt5
#ExportCSV2Mt5(NULL, "cTable1200_allEuroSymbols")
ExportCSV2Mt5 <- function(chosenSymbol, cTableName = "cTable")
{
  print(chosenSymbol)
  cTable <- readRDS(cTableName)
  if( !is.null( chosenSymbol) )
    cTable <- cTable[symbol==chosenSymbol] 
  #cTable <- cTable[index<=500]
  t<-dcast(cTable, clusterIndex ~ index, drop=FALSE)
  t[,clusterIndex:=NULL]
  fwrite(t, paste0(chosenSymbol, cTableName, ".csv"), col.names=FALSE, row.names = FALSE)
}
