library(RODBC)
library(randomForest)
library(dplyr)
library(pROC)

# 1. Get data
train_data <- read.csv("titanic_train_data")
score_data <- read.csv("titanic_score_data")
dim(train_data)
colnames(train_data)
str(train_data)
top_data <- head(train_data,6)


# 2. Extract variables
variables <- c("Pclass","Sex","Age","SibSp",      
               "Parch","Ticket","Fare","Embarked","Survived")
train <- train_data[variables]
variables <- c("Pclass","Sex","Age","SibSp",      
               "Parch","Ticket","Fare","Embarked")
score <- score_data[variables]


# 3. replace all NA - numerical
for(i in 1:ncol(train)){
  train[is.na(train[,i]), i] <- mean(train[,i], na.rm = TRUE)
}
for(i in 1:ncol(score)){
  score[is.na(score[,i]), i] <- mean(score[,i], na.rm = TRUE)
}


# 4. replace all NA - categorical
train %>%
  group_by(Embarked) %>%
  summarize(count =  n())
train[is.na(train[,8]),8] <- 'S'
any(is.na(train))

score %>%
  group_by(Embarked) %>%
  summarize(count =  n())
score[is.na(score[,8]),8] <- 'S'
any(is.na(score))

# 5. Split test-training 70:30
smp_size <- floor(0.70 * nrow(train))
set.seed(123)
train_ind <- sample(seq_len(nrow(train)), size = smp_size)
training <- train[train_ind, ]
testing <- train[-train_ind, ]


# 6. Random Forest
training$Survived <- as.factor(training$Survived)
set.seed(123)
rf_titanic <- randomForest(Survived~.,data=training)


# 7. Determine accruracy for testing set
predictions <- predict(rf_titanic, testing)
predictions <- as.numeric(levels(predictions)[predictions])
testing$Survived <- as.numeric(testing$Survived)
roc <- roc(testing$Survived, predictions)  
plot(roc, print.auc=TRUE)
roc

# 8. Predict on scoring
score_predictions <- predict(rf_titanic, score)

# export your prediction
# write.csv(score_predictions,"submission.csv",row.names=FALSE)
