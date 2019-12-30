library(e1071)
library(needs)
needs(readr,
      dplyr,
      ggplot2,
      corrplot,
      gridExtra,
      pROC,
      MASS,
      caTools,
      caret,
      caretEnsemble,
      doMC)
#registerDoMC(cores = 3)

data <- read.csv("data (1).csv")
str(data)

data$diagnosis <- as.factor(data$diagnosis)
# the 33 column is not right
data[,33] <- NULL
summary(data)

prop.table(table(data$diagnosis))

corr_mat <- cor(data[,3:ncol(data)])
corrplot(corr_mat, order = "hclust", tl.cex = 1, addrect = 8)

set.seed(1234)
data_index <- createDataPartition(data$diagnosis, p=0.7, list = FALSE)
train_data <- data[data_index, -1]
test_data <- data[-data_index, -1]

pca_res <- prcomp(data[,3:ncol(data)], center = TRUE, scale = TRUE)
plot(pca_res, type="l")

lda_res <- lda(diagnosis~., data, center = TRUE, scale = TRUE) 
lda_df <- predict(lda_res, data)$x %>% as.data.frame() %>% cbind(diagnosis=data$diagnosis)
lda_res

ggplot(lda_df, aes(x=LD1, y=0, col=diagnosis)) + geom_point(alpha=0.5)

ggplot(lda_df, aes(x=LD1, fill=diagnosis)) + geom_density(alpha=0.5)

train_data_lda <- lda_df[data_index,]
test_data_lda <- lda_df[-data_index,]

fitControl <- trainControl(method="cv",
                           number = 5,
                           preProcOptions = list(thresh = 0.99), # threshold for pca preprocess
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary)

model_lda <- train(diagnosis~.,
                   train_data_lda,
                   method="lda2",
                   #tuneLength = 10,
                   metric="ROC",
                   preProc = c("center", "scale"),
                   trControl=fitControl)
pred_lda <- predict(model_lda, test_data_lda)
cm_lda <- confusionMatrix(pred_lda, test_data_lda$diagnosis, positive = "M")
cm_lda

pred_prob_lda <- predict(model_lda, test_data_lda, type="prob")
#roc_lda <- roc(test_data_lda$diagnosis, pred_prob_lda$M)
#plot(roc_lda)
colAUC(pred_prob_lda, test_data_lda$diagnosis, plotROC=TRUE)

#RandomForest
model_rf <- train(diagnosis~.,
                  train_data,
                  method="ranger",
                  metric="ROC",
                  #tuneLength=10,
                  #tuneGrid = expand.grid(mtry = c(2, 3, 6)),
                  preProcess = c('center', 'scale'),
                  trControl=fitControl)
pred_rf <- predict(model_rf, test_data)
cm_rf <- confusionMatrix(pred_rf, test_data$diagnosis, positive = "M")
cm_rf

#Random forest with pca
model_pca_rf <- train(diagnosis~.,
                  train_data,
                  method="ranger",
                  metric="ROC",
                  #tuneLength=10,
                  #tuneGrid = expand.grid(mtry = c(2, 3, 6)),
                  preProcess = c('center', 'scale', 'pca'),
                  trControl=fitControl)
pred_pca_rf <- predict(model_pca_rf, test_data)
cm_pca_rf <- confusionMatrix(pred_pca_rf, test_data$diagnosis, positive = "M")
cm_pca_rf


#KNN
model_knn <- train(diagnosis~.,
                   train_data,
                   method="knn",
                   metric="ROC",
                   preProcess = c('center', 'scale'),
                   tuneLength=10,
                   trControl=fitControl)
pred_knn <- predict(model_knn, test_data)
cm_knn <- confusionMatrix(pred_knn, test_data$diagnosis, positive = "M")
cm_knn

pred_prob_knn <- predict(model_knn, test_data, type="prob")
roc_knn <- roc(test_data$diagnosis, pred_prob_knn$M)
plot(roc_knn)

#Neural Networks
model_nnet <- train(diagnosis~.,
                    train_data,
                    method="nnet",
                    metric="ROC",
                    preProcess=c('center', 'scale'),
                    trace=FALSE,
                    tuneLength=10,
                    trControl=fitControl)

pred_nnet <- predict(model_nnet, test_data)
cm_nnet <- confusionMatrix(pred_nnet, test_data$diagnosis, positive = "M")
cm_nnet

#Neural Networks with PCA
model_pca_nnet <- train(diagnosis~.,
                    train_data,
                    method="nnet",
                    metric="ROC",
                    preProcess=c('center', 'scale', 'pca'),
                    tuneLength=10,
                    trace=FALSE,
                    trControl=fitControl)

pred_pca_nnet <- predict(model_pca_nnet, test_data)
cm_pca_nnet <- confusionMatrix(pred_pca_nnet, test_data$diagnosis, positive = "M")
cm_pca_nnet

#Neural Networks with LDA
model_lda_nnet <- train(diagnosis~.,
                    train_data_lda,
                    method="nnet",
                    metric="ROC",
                    preProcess=c('center', 'scale'),
                    tuneLength=10,
                    trace=FALSE,
                    trControl=fitControl)
pred_lda_nnet <- predict(model_lda_nnet, test_data_lda)
cm_lda_nnet <- confusionMatrix(pred_lda_nnet, test_data_lda$diagnosis, positive = "M")
cm_lda_nnet

#SVM with Radial Kernal
model_svm <- train(diagnosis~.,
                    train_data,
                    method="svmRadial",
                    metric="ROC",
                    preProcess=c('center', 'scale'),
                    trace=FALSE,
                    trControl=fitControl)
pred_svm <- predict(model_svm, test_data)
cm_svm <- confusionMatrix(pred_svm, test_data$diagnosis, positive = "M")
cm_svm


#Naive Bayes
model_nb <- train(diagnosis~.,
                    train_data,
                    method="nb",
                    metric="ROC",
                    preProcess=c('center', 'scale'),
                    trace=FALSE,
                    trControl=fitControl)
pred_nb <- predict(model_nb, test_data)
cm_nb <- confusionMatrix(pred_nb, test_data$diagnosis, positive = "M")
cm_nb

#Naive Bayes (LDA)
model_lda_nb <- train(diagnosis~.,
                    train_data_lda,
                    method="nb",
                    metric="ROC",
                    preProcess=c('center', 'scale'),
                    trace=FALSE,
                    trControl=fitControl)

pred_lda_nb <- predict(model_lda_nb, test_data_lda)
cm_lda_nb <- confusionMatrix(pred_lda_nb, test_data$diagnosis, positive = "M")
cm_lda_nb

#Model result comparasion
model_list <- list(RF=model_rf, PCA_RF=model_pca_rf, 
                   NNET=model_nnet, PCA_NNET=model_pca_nnet, LDA_NNET=model_lda_nnet, 
                   KNN = model_knn, SVM=model_svm, NB=model_nb, LDA_NB=model_lda_nb)
resamples <- resamples(model_list)

#Correlation between models
model_cor <- modelCor(resamples)
corrplot(model_cor)
model_cor
bwplot(resamples, metric="ROC")

cm_list <- list(RF=cm_rf, PCA_RF=cm_pca_rf, 
                   NNET=cm_nnet, PCA_NNET=cm_pca_nnet, LDA_NNET=cm_lda_nnet, 
                   KNN = cm_knn, SVM=cm_svm, NB=cm_nb, LDA_NB=cm_lda_nb)

cm_list_results <- sapply(cm_list, function(x) x$byClass)
cm_list_results
cm_results_max <- apply(cm_list_results, 1, which.is.max)

 output_report <- data.frame(metric=names(cm_results_max), 
                            best_model=colnames(cm_list_results)[cm_results_max],
                            value=mapply(function(x,y) {cm_list_results[x,y]}, 
                            names(cm_results_max), 
                            cm_results_max))
rownames(output_report) <- NULL
output_report
