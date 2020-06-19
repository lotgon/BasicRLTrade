require(data.table)
require(rTTRatesHistory)
require(lubridate)

DownloadBars <- function(symbol, startDate, endDate, periodicity, type){
  server <- "ttlive.fxopen.com"
  login <- 100
  password <- "TTqfdeppmhDR"
  outputDir <- normalizePath(".//.cache")

  parameters <- sprintf("-a=%s -p=5042 -u=%s -w=%s -s=%s -f=%s -t=%s -o=csv -v=false -d=%s -l=\"%s\" -r=%s", 
                        server, login, password, symbol, startDate, endDate, periodicity, outputDir, type)
  system2(".//tools//QuotesDownloader//QuotesDownloader.exe", parameters, stdout = T)

}

#GetQuotes("AUDUSD", startDate=ISOdate(2016, 1, 1, tz = "UTC"), endDate=ISOdate(2019, 1, 1, tz = "UTC"), periodicity="M15", type="BidAsk")
GetQuotes <- function(symbol, startDate=ISOdate(2013, 1, 1, tz = "UTC"), endDate=ISOdate(2014, 1, 1, tz = "UTC"), periodicity="M15", type="Asks"){
  if( type == "BidAsk"){
    asks <- GetQuotes(symbol, startDate, endDate, periodicity, "Asks")
    bids <- GetQuotes(symbol, startDate, endDate, periodicity, "Bids")
    return(merge(asks, bids, by="datetime", suffixes = c(".ask", ".bid")))
  }
  qdFileName <- paste0(".//.cache//", symbol, " ", substr(type, 0, 3), " ", periodicity, "  ", format(startDate, "%Y%m%d"), "  ", format(endDate, "%Y%m%d"), ".csv")
  if( !file.exists(qdFileName) ){
    DownloadBars(symbol, startDate, endDate, periodicity, type)
  }
  d <- fread(qdFileName, col.names = c("datetime", "open", "close", "low", "high", "volume"))
  d[,datetime:=ymd_hms(datetime)]
  return(d)
}

#download all symbols
#DownloadAll() 
DownloadAll <- function(){
  #t<-"XNGUSD,#NDXm,#WS30m,#ESX50,#FCHI,#GDAXIm,#AUS200,#NDX,#SPXm,#WS30,#UK100,#GDAXI,#SPX,#J225,XBRUSD,XTIUSD,BTCCNH,BTCEUR,BTCRUB,BTCUSD,BTCGBP,BTCJPY,BCHUSD,BCHBTC,DSHBTC,DSHCNH,DSHUSD,EMCBTC,EMCUSD,EOSBTC,EOSEUR,EOSGBP,EOSJPY,EOSUSD,ETHBTC,ETHUSD,ETHLTC,ETHRUB,ETHEUR,ETHCNH,ETHJPY,MBTUSD,IOTBTC,LTCBTC,LTCCNH,LTCEUR,LTCRUB,LTCUSD,LTCJPY,NMCBTC,NMCUSD,XMRBTC,NEOBTC,ETCBTC,OMNBTC,PPCBTC,PPCUSD,XRPEUR,XRPUSD,XRPBTC,USTUSD,EURUSD,GBPUSD,AUDUSD,USDJPY,USDCAD,USDCNH,USDRUB,NZDUSD,USDCHF,AUDCAD,AUDCHF,AUDJPY,AUDNZD,CADJPY,CHFJPY,EURAUD,EURCAD,EURCHF,EURGBP,EURJPY,EURNZD,GBPAUD,GBPCHF,GBPJPY,NZDJPY,USDDKK,USDHKD,USDMXN,USDNOK,USDPLN,USDSEK,USDSGD,USDTRY,EURDKK,EURHKD,EURNOK,EURPLN,EURSEK,EURTRY,GBPCAD,GBPNZD,GBPSGD,CADCHF,HKDJPY,NOKJPY,NOKSEK,NZDCAD,NZDCHF,NZDSGD,SGDJPY,XAUUSD,XAGUSD"
  t<-"EURUSD,GBPUSD,AUDUSD,USDJPY,USDCAD,USDCNH,USDRUB,NZDUSD,USDCHF,AUDCAD,AUDCHF,AUDJPY,AUDNZD,CADJPY,CHFJPY,EURAUD,EURCAD,EURCHF,EURGBP,EURJPY,EURNZD,GBPAUD,GBPCHF,GBPJPY,NZDJPY,USDDKK,USDHKD,USDMXN,USDNOK,USDPLN,USDSEK,USDSGD,USDTRY,EURDKK,EURHKD,EURNOK,EURPLN,EURSEK,EURTRY,GBPCAD,GBPNZD,GBPSGD,CADCHF,HKDJPY,NOKJPY,NOKSEK,NZDCAD,NZDCHF,NZDSGD,SGDJPY,XAUUSD,XAGUSD"
  symbols <- sort(strsplit(t, ",")[[1]])
  for( symbol in symbols)
    GetQuotes(symbol, startDate=ISOdate(2012, 1, 1, tz = "UTC"), endDate=ISOdate(2018, 12, 1, tz = "UTC"))
}
ttsConn <- NULL
GetLastQuotes <- function( symbol, number){
  if( is.null(ttsConn))
    ttsConn <- ttInitialize("ttlivewebapi.fxopen.com")
  results <- data.table()
  date <- now("UTC")
  while(number > 0){
    newData <-  ttsConn$GetBarsHistory(gsub("/", "", symbol), barsType = "Bid", "M15", date, count=-number)
    if( nrow(newData) ==0 )
      stop("Can`t get data by unknown reason")
    number = number - newData[,.N]
    results <- rbind( newData, results)
    date <- newData[1, Timestamp]
  }
  results[,.(open=Open)]
}
