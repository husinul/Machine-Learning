
# http://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=236&DB_Short_Name=On-Time
origData <- read.csv('C:/Users/Arjun S Kumar/Downloads/dot.csv')
m<- nrow(origData)

print(m)
#print (origData)

airports <- c('ATL','LAX','ORD','DFW','JFK','SFO','CLT','LAS','PHX')

origData <- subset(origData, DEST %in% airports & ORIGIN %in% airports)

print(airports)
y<-nrow(origData)
#print(y)

print(head(origData,2))
print (tail(origData,2))
       
# final attribute is marked with x and values are NA so we can remove it
# to remove an attribute       
origData$X <- NULL

#in our data some feautures are almost same, correlation is used to ensure it

cor(origData[c("ORIGIN_AIRPORT_ID", "ORIGIN_AIRPORT_SEQ_ID")])
cor(origData[c("DEST_AIRPORT_ID", "DEST_AIRPORT_SEQ_ID")])

origData$ORIGIN_AIRPORT_SEQ_ID <- NULL
origData$DEST_AIRPORT_SEQ_ID <- NULL

## to check two attributes having same value
mismatched <-origData[origData$CARRIER != origData$UNIQUE_CARRIER,]
nrow(mismatched)

# if nrow equals zero indicates two attibutes sharing same values so we can drop one

origData$UNIQUE_CARRIER <- NULL

# molding the data

#remove the examples(rows) whose values contain NA or ""
onTimeData <- origData[!is.na(origData$ARR_DEL15)& origData$ARR_DEL15!="" & !is.na(origData$DEP_DEL15)& origData$DEP_DEL15 !="",]
nrow(onTimeData)
nrow(origData)

# cHANGE DATATYPES
onTimeData$DISTANCE <- as.integer(onTimeData$DISTANCE)
onTimeData$CANCELLED <- as.integer(onTimeData$CANCELLED)
onTimeData$DIVERTED <- as.integer(onTimeData$DIVERTED)
onTimeData$ARR_DEL15 <- as.factor(onTimeData$ARR_DEL15)
onTimeData$DEP_DEL15 <- as.factor(onTimeData$DEP_DEL15)
onTimeData$DEST_AIRPORT_ID <- as.factor(onTimeData$DEST_AIRPORT_ID)
onTimeData$ORIGIN_AIRPORT_ID <- as.factor(onTimeData$ORIGIN_AIRPORT_ID)
onTimeData$DAY_OF_WEEK<- as.factor(onTimeData$DAY_OF_WEEK)
onTimeData$DAY_OF_MONTH <- as.factor(onTimeData$DAY_OF_MONTH)
onTimeData$ORIGIN <- as.factor(onTimeData$ORIGIN)
onTimeData$DEST <- as.factor(onTimeData$DEST)
onTimeData$DEP_TIME_BLK <- as.factor(onTimeData$DEP_TIME_BLK)
onTimeData$CARRIER <- as.factor(onTimeData$CARRIER)

# COUNT HOW MANY TRUE /FALSE
tapply(onTimeData$ARR_DEL15,onTimeData$ARR_DEL15, length)








