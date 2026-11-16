getwd()
setwd("D://1116")

#############################################################################
## Ball Dataset Analysis: Linear Regression vs CART vs Random Forest
#############################################################################

# 1. 載入必要的套件
library(rpart)
library(rpart.plot)
library(randomForest)
library(caTools)
library(ggplot2)
library(gridExtra)

# 2. 讀取資料
# 請確保 ball.txt 放在你的工作目錄下
ball_data <- read.table("ball.txt", header = TRUE, stringsAsFactors = TRUE)

# 查看資料結構
dim(ball_data)
str(ball_data)
head(ball_data, n = 5)

# 處理遺漏值（確保 Salary 沒有 NA）
ball_data <- na.omit(ball_data)

# 3. 切分訓練資料與測試資料 (80% 訓練, 20% 測試)
set.seed(123)
split <- sample.split(ball_data$Salary, SplitRatio = 0.8)
train_data <- subset(ball_data, split == TRUE)
test_data <- subset(ball_data, split == FALSE)

dim(train_data)
dim(test_data)


########################################################################
## 模型一：多元線性迴歸 (Multiple Linear Regression)
########################################################################
lm.fit <- glm(Salary ~ Bat86 + Hit86 + Hr86 + Run86 + Rb86 + Wlk86 + Yrs + Batcr + Hitcr + Hrcr + Runcr + Rbcr + Wlkcr,
              family = gaussian(link = "identity"),
              data = train_data)

summary(lm.fit)

# 預測與評估
pred.lm <- predict(lm.fit, newdata = test_data)
mse.lm <- mean((pred.lm - test_data$Salary)^2)
rmse.lm <- sqrt(mse.lm)


########################################################################
## 模型二：決策樹迴歸 (Decision Tree Regression / CART)
########################################################################
tree.fit <- rpart(Salary ~ Bat86 + Hit86 + Hr86 + Run86 + Rb86 + Wlk86 + Yrs + Batcr + Hitcr + Hrcr + Runcr + Rbcr + Wlkcr,
                  method = "anova",
                  data = train_data)

# 繪製決策樹圖
windows()
rpart.plot(tree.fit, extra = 101, main = "Regression Tree: Ball Salary")

# 預測與評估
pred.tree <- predict(tree.fit, newdata = test_data)
mse.tree <- mean((pred.tree - test_data$Salary)^2)
rmse.tree <- sqrt(mse.tree)


########################################################################
## 模型三：隨機森林迴歸 (Random Forest Regression)
########################################################################
set.seed(123)
rf.fit <- randomForest(
  Salary ~ Bat86 + Hit86 + Hr86 + Run86 + Rb86 + Wlk86 + Yrs + Batcr + Hitcr + Hrcr + Runcr + Rbcr + Wlkcr,
  data = train_data,
  ntree = 500,
  mtry = 4,
  nodesize = 5,
  importance = TRUE
)

print(rf.fit)

# 變數重要性圖
windows()
varImpPlot(rf.fit, main = "Variable Importance: Ball Data")

# 預測與評估
pred.rf <- predict(rf.fit, newdata = test_data)
mse.rf <- mean((pred.rf - test_data$Salary)^2)
rmse.rf <- sqrt(mse.rf)


########################################################################
## 模型表現大對決 (Model Comparison)
########################################################################
results <- data.frame(
  Model = c("Linear Regression", "Decision Tree", "Random Forest"),
  MSE = c(mse.lm, mse.tree, mse.rf),
  RMSE = c(rmse.lm, rmse.tree, rmse.rf)
)

print(results)