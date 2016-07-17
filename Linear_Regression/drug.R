# load data
data <- read.csv('D:/Studies/ML_Course_Materials/Assignment/lab-1/drugdata.csv')
print (data)


# dependent variable
y<- data$Time
print (y)

# independent variable
x<- data$Drug.Rem.

print(x)

# to find h(x) we use lm function (linear model)

res <- lm(y~x)

# to get the weight vectors, intercept = x0 and x1= x
print (res)

plot (x,y,col ='Red', xlab= 'Drug.Rem', ylab='Time')

# to plot the h(x)
abline(res,col='Green')

#shows the predicted output of all examples
fitted(res)

#use to insert the input query 
new.df <- data.frame(x = c(1))

# to predict the y value for the x = 
output <- predict( res, new.df)
print (output)