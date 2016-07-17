# load data
data <- read.csv('D:/Studies/ML_Course_Materials/Assignment/lab-1/drugdata.csv')
#print (data)


# dependent variable
y<- data$Time
#print (y)

# independent variable
x<- data$Drug.Rem.
#print(x)

#append ones to the vector make it as (x0=1)
x <- cbind(1,x)
# initalize theta vector as zeros
theta<- c(0,0)
# Number of examples
m <- nrow(x)
#Calculate cost

# Set learning parameter
alpha <- 0.001

#Number of iterations

iterations <- 90000
# updating thetas using gradient update (batch gradient is used here)
for(i in 1:iterations)
{
theta[1] <- theta[1] - alpha * (1/m) * sum(((x%*%theta)- y))
theta[2] <- theta[2] - alpha * (1/m) * sum(((x%*%theta)- y)*x[,2])
cost[i] <- sum(((x%*%theta)- y)^2)/(2*m) # calculating cost

}
plot(x[,2],y)
abline(theta,col='red')

#plot(cost,type='line',col='blue',xlab='Iterations',ylab='Cost Function')

#Predict for value of drug rem= 1mg
predict1 <- c(1,1) %*% theta
#predict2 <- c(1,7) %*% theta

#print(cost)
print(predict1)
#print(predict2)