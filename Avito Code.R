#Appendix A: R Code
# Kaggle-competititon "Avito Context Ad Clicks"
# See https://www.kaggle.com/c/avito-context-ad-clicks
# In order to run this script on Kaggle-scripts I had to limit the number of entries to read
# from the database as well as to decrease the sample-size. With the full dataset from the database as well
# as a sample of 20 millions entries
install.packages("data.table")
install.packages("caret")
install.packages("RSQLite")
install.packages("pbkrtest")
install.packages("sqldf")
install.packages("Amelia")
install.packages("e1071") 
install.packages("doParallel")
install.packages("kernlab")
install.packages("pROC")
install.packages("DMwR")
#-------------------------------------------------------------------------------------------------------
library("DMwR")
library("caret")
library("kernlab")
library("doParallel")
library("pROC")
library("e1071")
library(sqldf)
library("data.table")
library("RSQLite")
library(pbkrtest)
library(Amelia)
library(ggplot2)
library("caret")
library("mlbench")
library("e1071")
library("ROCR")
#-------------------------------------------------------------------------------------------------------
# Prepare database 
#-------------------------------------------------------------------------------------------------------
db <- dbConnect(SQLite(), dbname="C:/Users/ggade/Documents/MIS 620/Avito/database.sqlite")
#dbListTables(db)
#-------------------------------------------------------------------------------------------------------
# Define constants to improve readability of large number
#-------------------------------------------------------------------------------------------------------
thousand <- 1000
million  <- thousand * thousand 
billion  <- thousand * million
#-------------------------------------------------------------------------------------------------------
# Runs the query, fetches the given number of entries and returns a data.table
#-------------------------------------------------------------------------------------------------------
fetch  <- function(db, query, n = 10000) {
  result <- dbSendQuery(db, query)
  data <- dbFetch(result, n)
  dbClearResult(result)
  return(as.data.table(data))
}
#-------------------------------------------------------------------------------------------------------
#Preparing to extract the records from the sqlite database into R
#-------------------------------------------------------------------------------------------------------
AdsInfo <- fetch(db, "select  * from AdsInfo", 400*thousand)
Category <- fetch(db, "select  * from Category", -1)
Location <- fetch(db, "select  * from Location", -1)
PhoneRequestsStream <- fetch(db, "select  * from PhoneRequestsStream", 400*thousand)
VisitsStream <- fetch(db, "select  * from VisitsStream", 400*thousand)
UserInfo <- fetch(db, "select  * from UserInfo", 400*thousand)
SearchInfo <- fetch(db, "select  * from SearchInfo", 400*thousand)
trainSearchStream <- fetch(db, "select  * from trainSearchStream", 400*thousand)
#-------------------------------------------------------------------------------------------------------
#Buiilding the entire relational schema for the database
#-------------------------------------------------------------------------------------------------------
temp1 <- sqldf("select a.*, b.Level as Category_Level from AdsInfo a left join Category b on a.CategoryID = b.CategoryID ")
temp2 <- sqldf("select a.*, b.Level as Category_Level from SearchInfo a left join Category b on a.CategoryID = b.CategoryID ")
temp3 <- sqldf("select a.*, b.Level as Location_Level from temp1 a left join Location b on a.LocationID = b.LocationID ")
temp4 <- sqldf("select a.*, b.Level as Location_Level from temp2 a left join Location b on a.LocationID = b.LocationID ")
temp5 <- sqldf("select a.*, b.IPID as PR_IPID from temp3 a left join PhoneRequestsStream b on a.AdID = b.AdID ")
temp6 <- sqldf("select a.*, b.IPID as PR_IPID from UserInfo a left join PhoneRequestsStream b on a.UserID = b.UserID ")
temp6 <- subset(temp6,select=c(1,2,3,4,5))
temp7 <- sqldf("select a.*, b.IPID as VS_IPID, b.ViewDate from temp6 a left join VisitsStream b on a.UserID = b.UserID ")
temp8 <- sqldf("select a.*, b.IPID as VS_IPID, b.ViewDate from temp5 a left join VisitsStream b on a.AdID = b.AdID ")
temp8 <- subset(temp8,select=c(1,2,3,5,7,8,9))
temp9 <- sqldf("select  a.*  from temp4 a left join temp7 b on a.UserID = b.UserID ")
temp10 <- sqldf("select a.*, b.SearchDate,  b.UserID, b.IsUserLoggedOn,b.SearchQuery, b.LocationID, b.CategoryID, b.SearchParams, b.Category_Level, b.Location_Level  from trainSearchStream a left join temp9  b on a.SearchID = b.SearchID ")
#removing that data where objectType != 3 so 246362 records left
final <- sqldf("select a.*, b.Price,  b.IsContext  from temp10 a left join temp8  b on a.AdID = b.AdID where a.ObjectType =3")
missmap(final, main = "Missing values vs. observed")
summary(final)
#removing Price as its not affect IsClick
final <- subset(final,select=c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17))
#removing search Params, Object Type and search query
final<- subset(final, select= c(1,2,3,5,6,7,8,9,11,12,14, 15, 16))
#-------------------------------------------------------------------------------------------------------
#Checking factor, numeric type for variables
#-------------------------------------------------------------------------------------------------------
#displaying the RESULTS IF VARIABLES ARE FACTOR OR NOT
a<-sapply(final,function(x)is.factor(x))
b<-sapply(final,function(x)is.numeric(x))
final$IsClick <- as.factor(final$IsClick)
contrasts(final$IsClick)
final$IsUserLoggedOn <- as.factor(final$IsUserLoggedOn)
contrasts(final$IsUserLoggedOn)
final$Location_Level <- as.factor(final$Location_Level)
contrasts(final$Location_Level)
final$Category_Level <- as.factor(final$Category_Level)
contrasts(final$Category_Level)
final<- subset(final, select= c(1,2,3,4,5,6,7,8,9,10,11,12)) 
#-------------------------------------------------------------------------------------------------------
#Register core backend, using 4 cores
#-------------------------------------------------------------------------------------------------------
cl <- makeCluster(4)
registerDoParallel(cl)
#-------------------------------------------------------------------------------------------------------
set.seed(123)
final$IsClick <- factor(final$IsClick)
final$IsUserLoggedOn <- factor(final$IsUserLoggedOn)
final$Category_Level <-factor(final$Category_Level)
final$Location_Level <- factor(final$Location_Level)
final$Position<- factor(final$Position)
yesnofactors <- factor(c("yes", "no")) 
levels(final$IsClick) <- make.names(levels(factor(yesnofactors)))
str(final)

#-------------------------------------------------------------------------------------------------------
# create a 80/20 partition
#-------------------------------------------------------------------------------------------------------

inTrain<-createDataPartition(y=final$IsClick, p=.8, list=FALSE)
nrow(inTrain)
final.train <- final[inTrain,]
summary(final.train)
final.test <- final[-inTrain,]
summary(final.test)

#-------------------------------------------------------------------------------------------------------
#use SMOTE to adjust for sampling
#-------------------------------------------------------------------------------------------------------
final.train.smote <- SMOTE(IsClick ~., final.train, perc.over =400, perc.under=150)
table(final.train.smote$IsClick)
prop.table(table(final.train.smote$IsClick))

#-------------------------------------------------------------------------------------------------------
#some parameters to control the sampling during parameter tuning and testing
#-------------------------------------------------------------------------------------------------------

ctrl <- trainControl(method="repeatedcv", repeats=3,
                     classProbs=TRUE,
                     #function used to measure performance
                     summaryFunction = twoClassSummary)
#twoClassSummary is built in function with ROC, Sensitivity and Specificity
#-------------------------------------------------------------------------------------------------------
#Decision Tree
#-------------------------------------------------------------------------------------------------------
modelLookup("rpart")
m.rpart <- train(IsClick ~ ., 
                 trControl = ctrl,
                 metric = "ROC", #using AUC to find best performing parameters
                 preProc = c("range", "nzv"), #scale from 0 to 1 and from columns with zero variance
                 data = final.train.smote, 
                 method = "rpart")
m.rpart
plot(m.rpart)
p.rpart <- predict(m.rpart,final.test)
confusionMatrix(p.rpart,final.test$IsClick)
roc_final<-roc(as.numeric(final.test$IsClick),as.numeric(p.rpart))
plot.roc(roc_final)
#-------------------------------------------------------------------------------------------------------
#Logistic Regression
#-------------------------------------------------------------------------------------------------------
modelLookup("glm")
m.smote.glm <- train(IsClick~ .,
                     trControl = ctrl,
                     metric = "ROC", #using AUC to find best performing parameters
                     preProc = c("range", "nzv"), #scale from 0 to 1 and from columns with zero variance
                     data = final.train.smote,
                     method = "glm")
m.smote.glm
p.smote.glm<- predict(m.smote.glm,final.test)
confusionMatrix(p.smote.glm,final.test$IsClick)
roc_final<-roc(as.numeric(final.test$IsClick),as.numeric(p.smote.glm))
plot.roc(roc_final)
#-------------------------------------------------------------------------------------------------------
# randomForest with SMOTE
#-------------------------------------------------------------------------------------------------------
install.packages("randomForest")
library ("randomForest")
modelLookup("rf")
m.rf <- randomForest(IsClick ~., data = final.train.smote, ntree=50,do.trace=2,replace=FALSE,verboseiter=FALSE)
m.rf
p.rf<- predict(m.rf,final.test)
confusionMatrix(p.rf,final.test$IsClick)
plot(roc(final.test$IsClick,as.numeric(p.rf)))
#-------------------------------------------------------------------------------------------------------
# use the NB classifier with Laplace smoothing
#-------------------------------------------------------------------------------------------------------
#building the model using naiveBayes on data pre-processed using smote and laplace
m.smote.naivebayes = naiveBayes(IsClick ~., data = final.train.smote , laplace=.01)
#displaying model statistics
m.smote.naivebayes
p.smote.nb<- predict(m.smote.naivebayes,final.test)
confusionMatrix_nb <- confusionMatrix(p.smote.nb,final.test$IsClick)
confusionMatrix_nb
#computing the ROC for the model: AUC = 0.549
roc_final_naivebayes<-roc(as.numeric(final.test$IsClick),as.numeric(p.smote.nb))
roc_final_naivebayes

#plotting the ROC
plot.roc(roc_final_naivebayes)
#-------------------------------------------------------------------------------------------------------
#Box Plot
#-------------------------------------------------------------------------------------------------------
rValues <- resamples(list(rpart=m.rpart, glm =m.smote.glm))
bwplot(rValues, metric="ROC",  horizontal=TRUE,
       col=c("red","blue"))
#-------------------------------------------------------------------------------------------------------

