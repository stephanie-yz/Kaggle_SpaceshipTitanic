######################################## Spaceship titanic ########################################
#################### Version Log ####################
### v1 - first attempt: ensemble model with glm, rf and knn; used mostly default settings
### v2 - tuned/optimised rf mtry parameter; bin age in 10 year bands
### v3 - train adaboost with trees (no tuning, parameter chosen approximately), 
###       added to ensemble and made it the tie breaker

#################### Load packages ####################
library(tidyverse)
library(ggplot2)
library(gridExtra)
library(caret)
library(randomForest)
library(adabag)
library(tictoc)
adapackageurl <- "https://cran.r-project.org/src/contrib/Archive/fastAdaboost/fastAdaboost_1.0.0.tar.gz"
install.packages(adapackageurl, repos=NULL, type="source")

#################### Load data ####################
wd <- 'C:/Users/HP/Documents/At University/Self-Education/R/Kaggle_Spaceship_Titanic'
setwd(wd)
list.files()

train_raw <- read.csv('train.csv')
test_raw <- read.csv('test.csv')


#################### Data cleaning - train ####################
str(train_raw)
summary(train_raw)

train_clean <- train_raw

### Remove NAs
sapply(train_clean, function(x) sum(is.na(x)))
  # amenities NAs = 0
train_clean <- train_clean %>% mutate(RoomService = replace_na(RoomService, 0), 
                                      FoodCourt = replace_na(FoodCourt, 0),
                                      ShoppingMall = replace_na(ShoppingMall, 0),
                                      Spa = replace_na(Spa, 0),
                                      VRDeck = replace_na(VRDeck, 0))
  #age NAs = avg age by HomePlanet
HomePlanet_avgage <- train_clean %>% group_by(HomePlanet) %>% summarise(Age = round(mean(na.omit(Age))))
train_clean <- train_clean %>% left_join(HomePlanet_avgage, by = 'HomePlanet', suffix = c('', '_hp')) %>% 
  mutate(Age = ifelse(is.na(Age), Age_hp, Age)) %>% 
  select(-Age_hp)



### Remove duplicates
sum(duplicated(train_clean$PassengerId))
sum(duplicated(train_clean$Name, incomparables = ''))
duplicated_names <- train_clean %>% 
  filter(Name %in% train_clean[duplicated(train_clean$Name, incomparables = ''), 'Name']) #no need to remove 


### Feature engineering
train_clean <- train_clean %>% mutate(TotalSpend = RoomService + FoodCourt + ShoppingMall + Spa + VRDeck)

train_clean <- train_clean %>% separate(Cabin, sep = '/', into = c('Deck', 'Num', 'Side')) %>% 
  mutate(Deck = ifelse(Deck == '', 'UNK', Deck), Num = replace_na(Num, '9999'), Side = replace_na(Side, 'UNK'))

  # age binning - since <10 seem to have higher success rate
train_clean <- train_clean %>% mutate(Age_bin = cut(Age, breaks = seq(min(Age), max(Age)+1, 10), include.lowest = T))


### Convert data types
sapply(train_clean, class)
numeric_cols <- c('Num', 'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpend')
factor_cols <- c('HomePlanet', 'CryoSleep', 'Deck', 'Side', 'Destination', 'VIP', 'Transported', 'Age_bin')
train_clean[numeric_cols] <- lapply(train_clean[numeric_cols], as.numeric)
train_clean[factor_cols] <- lapply(train_clean[factor_cols], as.factor)


str(train_clean)
summary(train_clean)



#################### EDA ####################


### HomePlace vs Detination
train_clean %>% select(HomePlanet, Destination) %>% map(table)
prop.table(table(train_clean$HomePlanet, train_clean$Destination), margin = 1)

### Histograms of continuous variables
names(train_clean)[sapply(train_clean, is.numeric)]

  #age histogram
train_clean %>% ggplot(aes(Age)) + geom_histogram(bins = 25, colour = 'blue')
train_clean %>% ggplot(aes(Age, colour = Transported)) + geom_density() +
  scale_x_continuous(breaks = seq(0, 80, 5)) + facet_wrap(vars(HomePlanet))

  #age boxplot
train_clean %>% ggplot(aes(HomePlanet, Age)) + geom_boxplot()
train_clean %>% ggplot(aes(Deck, Age)) + geom_boxplot()


  #amenities histogram
amenities_cols <- c("RoomService","FoodCourt","ShoppingMall","Spa","VRDeck","TotalSpend")
amenities_1way <- NULL
for (coln in amenities_cols) {
  amenities_1way[[coln]] <- train_clean %>% ggplot(aes_string(coln)) + 
    geom_histogram(aes(y=stat(count / sum(count))), bins = 20)
  amenities_1way[[paste(coln,"_nonzero")]] <- train_clean %>% ggplot(aes_string(coln)) + geom_density() + scale_x_log10()
}
gridExtra::grid.arrange(grobs = amenities_1way, ncol=2)

  #amenities histogram by deck
train_clean %>% ggplot(aes(TotalSpend)) + 
  geom_histogram(bins = 20) + scale_x_sqrt() + facet_wrap(vars(Deck))
deck_avgspend <- train_clean %>% group_by(VIP, Deck) %>% summarise(TotalSpend_avg = mean(TotalSpend), n = n())

### Correlation between amenities spend
train_clean %>% select(amenities_cols) %>% pairs
train_clean %>% select(amenities_cols) %>% na.omit() %>% cor()



#################### Model training ####################
train_input <- train_clean %>% select(-PassengerId, -Name, -TotalSpend, -Age)
train_input <- train_clean %>% select(HomePlanet, CryoSleep, Deck, Side, Destination, Transported)


### GLM ###
set.seed(123)
fit_glm_allvars <- train_input %>% train(Transported ~ ., data = ., method = 'glm')
fit_glm_allvars$finalModel
fit_glm_allvars$results


### Random Forest ###
set.seed(123)
tic()
fit_rf_allvars <- train_input %>% train(Transported ~ ., data = ., method = 'rf', ntree = 100,
                                        tuneGrid = data.frame(mtry = seq(3:9)))
toc()
confusionMatrix(fit_rf_allvars)
plot(fit_rf_allvars$finalModel)
varImpPlot(fit_rf_allvars$finalModel)
rf_tree_1 <- getTree(fit_rf_allvars$finalModel, 1, labelVar = T)


### KNN ###
set.seed(123)
tic()
fit_knn_allvars <- train_input %>% train(Transported ~ ., data = ., method = 'knn', 
                                         tuneGrid = data.frame(k = seq(40, 52, 3)))
toc()
confusionMatrix(fit_knn_allvars)
ggplot(fit_knn_allvars)


### AdaBoost - stumps ### (not used since takes too long to run with many stumps)
set.seed(123)
tic()
# fit_adab_allvars <- train_input %>% train(Transported ~ ., data = ., method = 'adaboost', 
#                                           tuneGrid = data.frame(nIter = seq(20, 60, 10), method = 'M1'))    
    #can't get train to work with adaboost or AdaBoost.M1 - just runs indefinitely
fit_adab_allvars <- train_input %>% boosting(Transported ~ ., data = ., mfinal = 15, control = rpart.control(maxdepth = 1))
toc()
confusionMatrix(factor(fit_adab_allvars$class), train_input$Transported)
plot(fit_adab_allvars$weights)


### AdaBoost - trees ###
set.seed(123)
tic()
fit_adab2_allvars <- train_input %>% adaboost(Transported ~ ., data = ., nIter = 5) #5 is randomly chosen to try avoid overfitting
  #fastAdaboost always send back full trees, rather than just stumps
toc()
confusionMatrix(predict(fit_adab2_allvars, train_input)$class, train_input$Transported)
plot(fit_adab2_allvars$weights)



### Ensemble - adaboost is tie breaker###
predict_ensb_train <- data.frame(glm = predict(fit_glm_allvars), rf = predict(fit_rf_allvars), 
                                 knn = predict(fit_knn_allvars), 
                                 adab_tree = predict(fit_adab2_allvars, train_input)$class)
# predict_ensb_train <- predict_ensb_train %>% mutate(maj_vote = ifelse(rowSums(. == 'False')/ncol(.) > 0.5, 'False', 'True'))
predict_ensb_train <- predict_ensb_train %>% mutate(maj_vote = case_when(rowSums(. == 'False')/ncol(.) == 0.5 ~ as.character(adab_tree),
                                                                         rowSums(. == 'False')/ncol(.) > 0.5 ~ 'False',
                                                                         TRUE ~ 'True'))
predict_ensb_train$maj_vote <- as.factor(predict_ensb_train$maj_vote)
confusionMatrix(predict_ensb_train$maj_vote, train_input$Transported)




#################### Data cleaning - test ####################
str(test_raw)
summary(test_raw)

test_clean <- test_raw

### Remove NAs
sapply(test_clean, function(x) sum(is.na(x)))
  # amenities NAs = 0
test_clean <- test_clean %>% mutate(RoomService = replace_na(RoomService, 0), 
                                      FoodCourt = replace_na(FoodCourt, 0),
                                      ShoppingMall = replace_na(ShoppingMall, 0),
                                      Spa = replace_na(Spa, 0),
                                      VRDeck = replace_na(VRDeck, 0))
  #age NAs = avg age by HomePlanet
test_clean <- test_clean %>% left_join(HomePlanet_avgage, by = 'HomePlanet', suffix = c('', '_hp')) %>% 
  mutate(Age = ifelse(is.na(Age), Age_hp, Age)) %>% 
  select(-Age_hp)


### Feature engineering
test_clean <- test_clean %>% mutate(TotalSpend = RoomService + FoodCourt + ShoppingMall + Spa + VRDeck)

test_clean <- test_clean %>% separate(Cabin, sep = '/', into = c('Deck', 'Num', 'Side')) %>% 
  mutate(Deck = ifelse(Deck == '', 'UNK', Deck), Num = replace_na(Num, '9999'), Side = replace_na(Side, 'UNK'))

  # age binning - since <10 seem to have higher success rate
test_clean <- test_clean %>% mutate(Age_bin = cut(Age, breaks = seq(min(Age), max(Age)+1, 10), include.lowest = T))


### Convert data types
sapply(test_clean, class)
numeric_cols <- c('Num', 'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpend')
factor_cols_test <- c('HomePlanet', 'CryoSleep', 'Deck', 'Side', 'Destination', 'VIP', 'Age_bin')
test_clean[numeric_cols] <- lapply(test_clean[numeric_cols], as.numeric)
test_clean[factor_cols_test] <- lapply(test_clean[factor_cols_test], as.factor)

str(test_clean)
summary(test_clean)



#################### Model prediction ####################
test_input <- test_clean %>% select(-PassengerId, -Name, -TotalSpend, -Age)

predict_glm_test <- predict(fit_glm_allvars, test_input)
predict_rf_test <- predict(fit_rf_allvars, test_input)
predict_knn_test <- predict(fit_knn_allvars, test_input)
predict_adab2_test <- predict(fit_adab2_allvars, test_input)$class



predict_ensb_test <- data.frame(glm = predict_glm_test, rf = predict_rf_test, knn = predict_knn_test, 
                                adab_tree = predict_adab2_test)
# predict_ensb_test <- predict_ensb_test %>% mutate(maj_vote = ifelse(rowSums(. == 'False')/ncol(.) > 0.5, 'False', 'True'))
predict_ensb_test <- predict_ensb_test %>% mutate(maj_vote = case_when(rowSums(. == 'False')/ncol(.) == 0.5 ~ as.character(adab_tree),
                                                                         rowSums(. == 'False')/ncol(.) > 0.5 ~ 'False',
                                                                         TRUE ~ 'True'))
predict_ensb_test$maj_vote <- as.factor(predict_ensb_test$maj_vote)

submission <- data.frame(PassengerId = test_clean$PassengerId, Transported = predict_ensb_test$maj_vote)
write.csv(submission, 'submissions_v3.csv', row.names = F, quote = F)

