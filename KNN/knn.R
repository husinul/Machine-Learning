library(gmodels)
library(class)

#Reading Glass data into a variable called data
data <- read.csv('C:/Downloads/orgdata.csv')
#print(data)

#Reading Glass traning data into a variable called tdata
tdata <- read.csv('C:/Downloads/traindata.csv')
#print(tdata)

#Reading Glass test data into a variable called testdata
testdata <- read.csv('C:/Downloads/testdata.csv')
#print(testdata)

table(data$Types_of_glass) # it helps us to get the numbers of data
#dataType <- factor(data$Type_of_glass, levels = c("1", "2", "3", "4", "5", "6", "7"), labels = c("building_windows_float_processed", "building_windows_non_float_processed", "vehicle_windows_float_processed", "vehicle_windows_non_float_processed (none in this database)", "containers", "tableware", "headlamps"))

# It gives the result in the percentage form rounded of to 1 decimal place( and so it's digits = 1)
round(prop.table(table(data$Types_of_glass)) * 100, digits = 2)  



#Normalizing function
normalize <- function(x) {return ((x - min(x)) / (max(x) - min(x))) }#normaalization function
tdata_n <- as.data.frame(lapply(tdata[1:9], normalize))
#print(tdata_n)

data_train_labels <- tdata[1:149,10]   #Stores output of training data in a variable
data_test_labels <- testdata[1:65,10] #Stores desired output of test data in a variable
print(data_train_labels)
print(data_test_labels)

#KNN function call
data_test_pred <- knn(train = tdata, test = testdata,cl = data_train_labels, k=7)

#Confusion Matrix
CrossTable(x=data_test_labels,y=data_test_pred,prop.chisq=FALSE)
