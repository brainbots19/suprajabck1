ggscreeplot <- function(pcobj, type = c('pev', 'cev')) 
{
  type <- match.arg(type)
  d <- pcobj$sdev^2
  yvar <- switch(type, 
                 pev = d / sum(d), 
                 cev = cumsum(d) / sum(d))
  yvar.lab <- switch(type,
                     pev = 'proportion of explained variance',
                     cev = 'cumulative proportion of explained variance')
  df <- data.frame(PC = 1:length(d), yvar = yvar)
  ggplot(data = df, aes(x = PC, y = yvar)) + 
    xlab('principal component number') + ylab(yvar.lab) +
    geom_point() + geom_path()
}

ggbiplot <- function(pcobj, choices = 1:2, scale = 1, pc.biplot = TRUE, 
                      obs.scale = 1 - scale, var.scale = scale, 
                      groups = NULL, ellipse = FALSE, ellipse.prob = 0.68, 
                      labels = NULL, labels.size = 3, alpha = 1, 
                      var.axes = TRUE, 
                      circle = FALSE, circle.prob = 0.69, 
                      varname.size = 3, varname.adjust = 1.5, 
                      varname.abbrev = FALSE, ...)
{
  library(ggplot2)
  library(plyr)
  library(scales)
  library(grid)
  stopifnot(length(choices) == 2)
  # Recover the SVD
  if(inherits(pcobj, 'prcomp')){
    nobs.factor <- sqrt(nrow(pcobj$x) - 1)
    d <- pcobj$sdev
    u <- sweep(pcobj$x, 2, 1 / (d * nobs.factor), FUN = '*')
    v <- pcobj$rotation
  } else if(inherits(pcobj, 'princomp')) {
    nobs.factor <- sqrt(pcobj$n.obs)
    d <- pcobj$sdev
    u <- sweep(pcobj$scores, 2, 1 / (d * nobs.factor), FUN = '*')
    v <- pcobj$loadings
  } else if(inherits(pcobj, 'PCA')) {
    nobs.factor <- sqrt(nrow(pcobj$call$X))
    d <- unlist(sqrt(pcobj$eig)[1])
    u <- sweep(pcobj$ind$coord, 2, 1 / (d * nobs.factor), FUN = '*')
    v <- sweep(pcobj$var$coord,2,sqrt(pcobj$eig[1:ncol(pcobj$var$coord),1]),FUN="/")
  } else if(inherits(pcobj, "lda")) {
      nobs.factor <- sqrt(pcobj$N)
      d <- pcobj$svd
      u <- predict(pcobj)$x/nobs.factor
      v <- pcobj$scaling
      d.total <- sum(d^2)
  } else {
    stop('Expected a object of class prcomp, princomp, PCA, or lda')
  }
  # Scores
  choices <- pmin(choices, ncol(u))
  df.u <- as.data.frame(sweep(u[,choices], 2, d[choices]^obs.scale, FUN='*'))
  # Directions
  v <- sweep(v, 2, d^var.scale, FUN='*')
  df.v <- as.data.frame(v[, choices])
  names(df.u) <- c('xvar', 'yvar')
  names(df.v) <- names(df.u)
  if(pc.biplot) {
    df.u <- df.u * nobs.factor
  }
  # Scale the radius of the correlation circle so that it corresponds to 
  # a data ellipse for the standardized PC scores
  r <- sqrt(qchisq(circle.prob, df = 2)) * prod(colMeans(df.u^2))^(1/4)
  # Scale directions
  v.scale <- rowSums(v^2)
  df.v <- r * df.v / sqrt(max(v.scale))
  # Change the labels for the axes
  if(obs.scale == 0) {
    u.axis.labs <- paste('standardized PC', choices, sep='')
  } else {
    u.axis.labs <- paste('PC', choices, sep='')
  }
  # Append the proportion of explained variance to the axis labels
  u.axis.labs <- paste(u.axis.labs, 
                       sprintf('(%0.1f%% explained var.)', 
                               100 * pcobj$sdev[choices]^2/sum(pcobj$sdev^2)))
  # Score Labels
  if(!is.null(labels)) {
    df.u$labels <- labels
  }
  # Grouping variable
  if(!is.null(groups)) {
    df.u$groups <- groups
  }
  # Variable Names
  if(varname.abbrev) {
    df.v$varname <- abbreviate(rownames(v))
  } else {
    df.v$varname <- rownames(v)
  }
  # Variables for text label placement
  df.v$angle <- with(df.v, (180/pi) * atan(yvar / xvar))
  df.v$hjust = with(df.v, (1 - varname.adjust * sign(xvar)) / 2)
  # Base plot
  g <- ggplot(data = df.u, aes(x = xvar, y = yvar)) + 
          xlab(u.axis.labs[1]) + ylab(u.axis.labs[2]) + coord_equal()
  if(var.axes) {
    # Draw circle
    if(circle) 
    {
      theta <- c(seq(-pi, pi, length = 50), seq(pi, -pi, length = 50))
      circle <- data.frame(xvar = r * cos(theta), yvar = r * sin(theta))
      g <- g + geom_path(data = circle, color = muted('white'), 
                         size = 1/2, alpha = 1/3)
    }
    # Draw directions
    g <- g +
      geom_segment(data = df.v,
                   aes(x = 0, y = 0, xend = xvar, yend = yvar),
                   arrow = arrow(length = unit(1/2, 'picas')), 
                   color = muted('red'))
  }
  # Draw either labels or points
  if(!is.null(df.u$labels)) {
    if(!is.null(df.u$groups)) {
      g <- g + geom_text(aes(label = labels, color = groups), 
                         size = labels.size)
    } else {
      g <- g + geom_text(aes(label = labels), size = labels.size)      
    }
  } else {
    if(!is.null(df.u$groups)) {
      g <- g + geom_point(aes(color = groups), alpha = alpha)
    } else {
      g <- g + geom_point(alpha = alpha)      
    }
  }
  # Overlay a concentration ellipse if there are groups
  if(!is.null(df.u$groups) && ellipse) {
    theta <- c(seq(-pi, pi, length = 50), seq(pi, -pi, length = 50))
    circle <- cbind(cos(theta), sin(theta))
    ell <- ddply(df.u, 'groups', function(x) {
      if(nrow(x) <= 2) {
        return(NULL)
      }
      sigma <- var(cbind(x$xvar, x$yvar))
      mu <- c(mean(x$xvar), mean(x$yvar))
      ed <- sqrt(qchisq(ellipse.prob, df = 2))
      data.frame(sweep(circle %*% chol(sigma) * ed, 2, mu, FUN = '+'), 
                 groups = x$groups[1])
    })
    names(ell)[1:2] <- c('xvar', 'yvar')
    g <- g + geom_path(data = ell, aes(color = groups, group = groups))
  }
  # Label the variable axes
  if(var.axes) {
    g <- g + 
    geom_text(data = df.v, 
              aes(label = varname, x = xvar, y = yvar, 
                  angle = angle, hjust = hjust), 
              color = 'darkred', size = varname.size)
  }
  # Change the name of the legend for groups
  # if(!is.null(groups)) {
  #   g <- g + scale_color_brewer(name = deparse(substitute(groups)), 
  #                               palette = 'Dark2')
  # }
  # TODO: Add a second set of axes
  return(g)
}

boxplot2g = function(x,y=NULL, groups = NULL, smooth = loess, smooth.args = list(span = 0.1),  colv = NULL, alpha = 1, n = 360,...){
  prbs <- c(0.25,0.5,0.75)
  if(is.null(y)){
    stopifnot(ncol(x)==2)   
    data <- as.data.frame(x)
  }else{
    data <- as.data.frame(cbind(x,y))   
  }
  if(is.null(groups)){
    data$groups <- as.factor(0)
  }else{
    data$groups <- as.factor(groups)
  }
  labs <- names(data)
  names(data) <- c("x","y","groups")
  DM <- data.matrix(data)
  #require(ggplot2)
  # initiate the smoother
  if(is.logical(smooth)){
    do.smooth <- smooth 
  }else{
    do.smooth <- TRUE   
  }
  if(do.smooth){
    poss.args <- names(formals(smooth))
    spec.args <- names(smooth.args)

    ind <- match(spec.args, poss.args)
    for(i in seq_along(ind)){
      formals(smooth)[ind[i]] <- smooth.args[[i]]   
    }   
    if("span" %in% poss.args){
      formals(smooth)$span <- formals(smooth)$span/3
    }
  }else{
    smooth <- NULL
  }
  phi = seq(360/n, 360, 360/n)/180*pi
  e1 <- new.env()
  e1$vectors <- cbind(sin(phi),cos(phi))
  ntv <- nlevels(data$groups)
  if(is.null(colv)){
    #print(ntv)
    if(ntv == 1){
      colv = 1  
    }else{
      colv <- rainbow(ntv)  
    }
  }
  e1$colv <- colv
  e1$lvls <- levels(data$groups)
  #colv <- colv[match(groups,levels(as.factor(data$groups)))]
  #e1$gp <- qplot(data$x, data$y, colour = data$groups) 
  e1$gp <- ggplot(data=data,aes(x=x,y=y,colour=groups))+geom_point(alpha=alpha) 
  #print(formals(smooth))
  if(ntv == 1){
    groupbox2d(x=data,env=e1,prbs=prbs,smooth=smooth,do.smooth)
  }else{
    by(data,groups, groupbox2d, env= e1, prbs = prbs, smooth = smooth)
  }
  #e1$gp <- e1$gp  + opts(legend.position = "none") 
  return(e1$gp)
}
groupbox2d = function( x, env, prbs, past, smooth){
  grp <- x[1,3] 
  colid <- match(grp, env$lvls)
  if(any(colid)){
    colv <- env$colv[]
  }else{
    colv <- env$col[1]  
  }
  xs <- x[,1:2]
  mm <- apply(xs,2,mean)
  xs <-  data.matrix(xs) - rep(mm,each=nrow(xs))
  S <- cov(xs)
  if (requireNamespace("MASS", quietly = TRUE)) {
    Sinv <- MASS::ginv(S)
    SSinv <- svd(Sinv)
    SSinv <- SSinv$u %*% diag(sqrt(SSinv$d))
    SS <- MASS::ginv(SSinv)
  }else{
    Sinv <- solve(S)
    SSinv <- svd(Sinv)
    SSinv <- SSinv$u %*% diag(sqrt(SSinv$d))
    SS <- solve(SSinv)  
  }
  xs <- xs %*% SSinv
  prj <- xs %*% t(env$vectors)
  qut <- t(apply(prj,2, function(z){
    quarts <- quantile(z, probs = prbs)
    iqr <- quarts[3]-quarts[1]
    w1 <- min(z[which(z >= quarts[1] - 1.5*iqr)])
    #w2 <- max(z[which(z <= quarts[3] + 1.5*iqr)])
    #return(c(w1,quarts,w2))
    return(c(w1,quarts))
  }))
  #print(formals(smooth))
  if( !is.null(smooth) ){
    n <- nrow(qut)
    qut <- apply(qut,2,function(z){
      x <- 1:(3*n)
      z <- rep(z,3)
      ys <- predict(smooth(z~x))
      return(ys[(n+1):(2*n)])
    })
    #print(dim(qut))
  }
  ccBox <- env$vectors*qut[,2]
  md <- data.frame((env$vectors*qut[,3])%*%SS)
  md <- sapply(md,mean)+mm      
  md[3] <- grp
  ccWsk <- env$vectors*qut[,1]
  ccc <- data.frame(rbind(ccBox,ccWsk) %*% SS + rep(mm,each=2*nrow(ccBox)))
  ccc$grp <- as.factor(rep(c("box","wsk"),each=nrow(ccBox)))
  ccc$groups <- factor(grp)
  md <- data.frame(md[1],md[2],grp)
  names(md) <- names(ccc)[-3]
  X1 <- NULL
  X2 <- NULL
  groups <- NULL
  #env$gp <- env$gp + geom_point(x=md[1],y=md[2],colour=md[3])
  env$gp <- env$gp + geom_point(data=md,aes(x=X1,y=X2, colour = groups),size=5) +  scale_colour_manual(values = colv) 
  env$gp <- env$gp + geom_path(data=ccc, aes(x=X1,y=X2,group=grp, colour = groups), alpha = 1/8)
  env$gp <- env$gp + geom_polygon(data=ccc,aes(x=X1,y=X2,group=grp, colour = groups, fill = groups), alpha = 1/8)
  env$gp <- env$gp + geom_point(data=md,aes(x=X1,y=X2),size=3,alpha=1,colour="white")
  env$gp <- env$gp + geom_point(data=md,aes(x=X1,y=X2),size=1,alpha=1)
  return( invisible(TRUE) )
}

raw.data <- read.csv("../input/data.csv")
print(sprintf("Number of data rows: %d",nrow(raw.data)))

print(sprintf("Number of data columns: %d",ncol(raw.data)))

knitr::kable(head(raw.data,5),caption="Raw data (first 5 rows)")

glimpse(raw.data)

summary(raw.data)

diagnostic <- plyr::count(raw.data$diagnosis)
print(sprintf("Malignant: %d | Benign: %d",diagnostic$freq[2],diagnostic$freq[1]))

print(sprintf("Percent of malignant tumor: %.1f%%",round(diagnostic$freq[2]/nrow(raw.data)*100,1)))

newNames = c(
  "fractal_dimension_mean",  "fractal_dimension_se", "fractal_dimension_worst",
  "symmetry_mean", "symmetry_se", "symmetry_worst",
  "concave.points_mean", "concave.points_se", "concave.points_worst",
  "concavity_mean","concavity_se", "concavity_worst",
  "compactness_mean", "compactness_se", "compactness_worst",
  "smoothness_mean", "smoothness_se", "smoothness_worst",
  "area_mean", "area_se", "area_worst",
  "perimeter_mean",  "perimeter_se", "perimeter_worst",
  "texture_mean" , "texture_se", "texture_worst",
  "radius_mean", "radius_se", "radius_worst"
)

bc.data = (raw.data[,newNames])
bc.diag = raw.data[,2]

scales <- list(x=list(relation="free"),y=list(relation="free"), cex=0.6)
featurePlot(x=bc.data, y=bc.diag, plot="density",scales=scales,
            layout = c(3,10), auto.key = list(columns = 2), pch = "|")
            
newNamesMean = c(
  "fractal_dimension_mean",
  "symmetry_mean",
  "concave.points_mean", 
  "concavity_mean",
  "compactness_mean",
  "smoothness_mean", 
  "area_mean",
  "perimeter_mean",
  "texture_mean" ,
  "radius_mean"
)


bcM.data = (raw.data[,newNamesMean])
bcM.diag = raw.data[,2]
scales <- list(x=list(relation="free"),y=list(relation="free"), cex=0.4)
featurePlot(x=bcM.data, y=bcM.diag, plot="pairs",scales=scales,
         auto.key = list(columns = 2), pch=".")
         
newNamesSE = c(
  "fractal_dimension_se",
  "symmetry_se",
  "concave.points_se", 
  "concavity_se",
  "compactness_se",
  "smoothness_se", 
  "area_se",
  "perimeter_se",
  "texture_se" ,
  "radius_se"
)

bcSE.data = (raw.data[,newNamesSE])
bcSE.diag = raw.data[,2]
scales <- list(x=list(relation="free"),y=list(relation="free"), cex=0.4)
featurePlot(x=bcSE.data, y=bcSE.diag, plot="pairs",scales=scales,
         auto.key = list(columns = 2), pch=".")
         
newNamesW = c(
  "fractal_dimension_worst",
  "symmetry_worst",
  "concave.points_worst", 
  "concavity_worst",
  "compactness_worst",
  "smoothness_worst", 
  "area_worst",
  "perimeter_worst",
  "texture_worst" ,
  "radius_worst"
)

bcW.data = (raw.data[,newNamesW])
bcW.diag = raw.data[,2]
scales <- list(x=list(relation="free"),y=list(relation="free"), cex=0.4)
featurePlot(x=bcW.data, y=bcW.diag, plot="pairs",scales=scales,
         auto.key = list(columns = 2), pch=".")
         
 nc=ncol(raw.data)
dfm <- raw.data[raw.data$diagnosis=='M',4:nc-1]

m <- data.matrix(dfm)
library(RColorBrewer)
cls = colorRampPalette(brewer.pal(8, "Dark2"))(256)
heatmap(m, scale="column", col = cls, labRow=FALSE,Colv=NA, Rowv=NA)

nc=ncol(raw.data)
dfb <- raw.data[raw.data$diagnosis=='B',4:nc-1]

m <- data.matrix(dfb)
library(RColorBrewer)
cls = colorRampPalette(brewer.pal(8, "Dark2"))(256)
heatmap(m, scale="column", col = cls, labRow = FALSE, Colv=NA, Rowv=NA)

nc=ncol(raw.data)
df <- raw.data[,3:nc-1]
df$diagnosis <- as.integer(factor(df$diagnosis))-1
correlations <- cor(df,method="pearson")
corrplot(correlations, number.cex = .9, method = "square", 
         hclust.method = "ward", order = "FPC",
         type = "full", tl.cex=0.8,tl.col = "black")

b1 <- boxplot2g(bc.data$radius_worst, bc.data$perimeter_mean, bc.diag, smooth = loess, NULL, NULL) +
  labs(title="Boxplot 2G for highly correlated features", subtitle = "Perimeter mean vs. Radius worst", x="Radius worst", y="Perimeter mean") + theme_bw()
b2 <- boxplot2g(bc.data$area_worst, bc.data$radius_worst, bc.diag, smooth = loess, NULL, NULL) +
  labs(title="Boxplot 2G for highly correlated features", subtitle = "Area worst vs. Radius worst", x="Radius worst", y="Area worst") + theme_bw()
b3 <- boxplot2g(bc.data$texture_mean, bc.data$texture_worst, bc.diag, smooth = loess, NULL, NULL) +
  labs(title="Boxplot 2G for highly correlated features", subtitle = "Texture mean vs. Texture worst", x="Texture worst", y="Texture mean") + theme_bw()
b4 <- boxplot2g(bc.data$area_worst, bc.data$perimeter_mean, bc.diag, smooth = loess, NULL, NULL) +
  labs(title="Boxplot 2G for highly correlated features", subtitle = "Perimeter mean vs. Area worst", x="Area worst", y="Perimeter mean") + theme_bw()
grid.arrange(b1, b2, b3, b4, ncol=2)

b5 <- boxplot2g(bc.data$radius_mean, bc.data$fractal_dimension_mean, bc.diag, smooth = loess, NULL, NULL) +
  labs(title="Boxplot 2G for inverse correlated feat.", subtitle = "Fractal dimension mean vs. Radius mean", x="Radius mean", y="Fractal dimension mean") + theme_bw()
b6 <- boxplot2g(bc.data$area_mean, bc.data$fractal_dimension_mean, bc.diag, smooth = loess, NULL, NULL) +
  labs(title="Boxplot 2G for inverse correlated feat.", subtitle = "Fractal dimension mean vs. Area mean", x="Area mean", y="Fractal dimension mean") + theme_bw()
grid.arrange(b5, b6, ncol=2)

b9 <- boxplot2g(bc.data$fractal_dimension_worst, bc.data$area_se, bc.diag, smooth = loess, NULL, NULL) +
  labs(title="Boxplot 2G for low correlated features", subtitle = "Area SE vs. Fractal dimmension worst", x="Fractal dimmension worst", y="Area SE") + theme_bw()
b10 <- boxplot2g(bc.data$fractal_dimension_worst, bc.data$radius_se, bc.diag, smooth = loess, NULL, NULL) +
  labs(title="Boxplot 2G for low correlated features", subtitle = "Radius SE vs. Fractal dimmension worst", x="Fractal dimmension worst", y="Radius SE") + theme_bw()
b11 <- boxplot2g(bc.data$texture_mean, bc.data$smoothness_mean, bc.diag, smooth = loess, NULL, NULL) +
  labs(title="Boxplot 2G for low correlated features", subtitle = "Smoothness mean vs. Texture mean", x="Texture mean", y="Smoothness mean") + theme_bw()
b12 <- boxplot2g(bc.data$perimeter_worst, bc.data$fractal_dimension_se, bc.diag, smooth = loess, NULL, NULL) +
  labs(title="Boxplot 2G for low correlated features", subtitle = "Fractal dimmension SE vs. Perimeter worst", x="Perimeter worst", y="Fractal dimension SE") + theme_bw()
grid.arrange(b9, b10, b11, b12, ncol=2)


bc.pca <- prcomp(bc.data, center=TRUE, scale.=TRUE)
plot(bc.pca, type="l", main='',color='red')

grid(nx = 10, ny = 14)
title(main = "Principal components weight", sub = NULL, xlab = "Components")
box()

ggbiplot(bc.pca, choices=1:2, obs.scale = 1, var.scale = 1, groups = bc.diag, 
  ellipse = TRUE, circle = TRUE, varname.size = 3, ellipse.prob = 0.68, circle.prob = 0.69) +
  scale_color_discrete(name = 'Diagnosis (B: beningn, M: malignant)') + theme_bw() + 
  labs(title = "Principal Component Analysis", 
  subtitle = "1. Data distribution in the plan of PC1 and PC2\n2. Directions of components in the same plane") +
  theme(legend.direction = 'horizontal', legend.position = 'bottom')
  

pc34<- ggbiplot(bc.pca, choices=3:4, obs.scale = 1, var.scale = 1, groups = bc.diag, 
        ellipse = TRUE, circle = TRUE, varname.size = 3, ellipse.prob = 0.68, circle.prob = 0.69) +
        scale_color_discrete(name = 'Diagnosis (B: beningn, M: malignant)') + theme_bw() + 
        labs(title = "Principal Component Analysis", 
        subtitle = "1. Data distribution in the plan of PC3 and PC4\n2. Directions of components in the same plane") +
        theme(legend.direction = 'horizontal', legend.position = 'bottom')

pc56<- ggbiplot(bc.pca, choices=5:6, obs.scale = 1, var.scale = 1, groups = bc.diag, 
        ellipse = TRUE, circle = TRUE, varname.size = 3, ellipse.prob = 0.68, circle.prob = 0.69) +
        scale_color_discrete(name = 'Diagnosis (B: beningn, M: malignant)') + theme_bw() + 
        labs(title = "Principal Component Analysis", 
        subtitle = "1. Data distribution in the plan of PC5 and PC6\n2. Directions of components in the same plane") +
        theme(legend.direction = 'horizontal', legend.position = 'bottom')
grid.arrange(pc34, pc56, ncol=2)


library(Rtsne)
colors = rainbow(length(unique(bc.diag)))
names(colors) = unique(bc.diag)
set.seed(31452)

tsne <- Rtsne(bc.data, dims=2, perplexity=30, 
              verbose=TRUE, pca=TRUE, 
              theta=0.01, max_iter=1000)
              
plot(tsne$Y, t='n', main="t-Distributed Stochastic Neighbor Embedding (t-SNE)",
     xlab="t-SNE 1st dimm.", ylab="t-SNE 2nd dimm.")
text(tsne$Y, labels=bc.diag, cex=0.5, col=colors[bc.diag])

df <- raw.data[,2:32]
df$diagnosis = as.integer(factor(df$diagnosis))-1
nrows <- nrow(df)
set.seed(314)
indexT <- sample(1:nrow(df), 0.7 * nrows)
#separate train and validation set
trainset = df[indexT,]
testset =   df[-indexT,]
n <- names(trainset)

plot(trainset.rf, main="Random Forest: MSE error vs. no of trees")


varimp <- data.frame(trainset.rf$importance)
  vi1 <- ggplot(varimp, aes(x=reorder(rownames(varimp),IncNodePurity), y=IncNodePurity)) +
  geom_bar(stat="identity", fill="tomato", colour="black") +
  coord_flip() + theme_bw(base_size = 8) +
  labs(title="Prediction using RandomForest with 500 trees", subtitle="Variable importance (IncNodePurity)", x="Variable", y="Variable importance (IncNodePurity)")
  vi2 <- ggplot(varimp, aes(x=reorder(rownames(varimp),X.IncMSE), y=X.IncMSE)) +
  geom_bar(stat="identity", fill="lightblue", colour="black") +
  coord_flip() + theme_bw(base_size = 8) +
  labs(title="Prediction using RandomForest with 500 trees", subtitle="Variable importance (%IncMSE)", x="Variable", y="Variable importance (%IncMSE)")
grid.arrange(vi1, vi2, ncol=2)


testset$predicted <- round(predict(trainset.rf ,testset),0)

plotConfusionMatrix <- function(testset, sSubtitle) {
    tst <- data.frame(testset$predicted, testset$diagnosis)
    opts <- c("Predicted", "True")
    names(tst) <- opts
    cf <- plyr::count(tst)
    cf[opts][cf[opts]==0] <- "Benign"
    cf[opts][cf[opts]==1] <- "Malignant"
    ggplot(data =  cf, mapping = aes(x = True, y = Predicted)) +
      labs(title = "Confusion matrix", subtitle = sSubtitle) +
      geom_tile(aes(fill = freq), colour = "grey") +
      geom_text(aes(label = sprintf("%1.0f", freq)), vjust = 1) +
      scale_fill_gradient(low = "gold", high = "tomato") +
      theme_bw() + theme(legend.position = "none")
}
plotConfusionMatrix(testset,"Prediction using RandomForest with 500 trees")

print(sprintf("Area under curve (AUC) : %.3f",auc(testset$diagnosis, testset$predicted)))

predicted_rf = testset$predicted

features_list = c("perimeter_worst", "area_worst", "concave.points_worst", "radius_worst", 
                  "concavity_mean", "concavity_worst","area_se", "concave.points_mean",
                  "texture_worst", "area_mean", "texture_mean", "area_mean", 
                  "radius_mean", "radius_se", "perimeter_mean", "perimeter_se",
                  "compactness_worst", "smoothness_worst", "concavity_se",
                  "fractal_dimension_worst", "symmetry_worst",  "diagnosis")
#define train and validation set
trainset_fl = trainset[,features_list]
testset_fl =   testset[,features_list]
#training
n <- names(trainset_fl)
rf.form <- as.formula(paste("diagnosis ~", paste(n[!n %in% "diagnosis"], collapse = " + ")))
trainset.rf <- randomForest(rf.form,trainset_fl,ntree=500,importance=T)


#prediction
testset_fl$predicted <- round(predict(trainset.rf ,testset_fl),0)

plotConfusionMatrix(testset_fl,"Prediction using RandomForest with reduced features set")

print(sprintf("Area under curve (AUC) : %.3f",auc(testset_fl$diagnosis, testset_fl$predicted)))

n<-names(trainset)
gbm.form <- as.formula(paste("diagnosis ~", paste(n[!n %in% "diagnosis"], collapse = " + ")))
gbmCV = gbm(formula = gbm.form,
               distribution = "bernoulli",
               data = trainset,
               n.trees = 500,
               shrinkage = .1,
               n.minobsinnode = 15,
               cv.folds = 5,
               n.cores = 1)
               
 gbmTest = predict(object = gbmCV,
                           newdata = testset,
                           n.trees = optimalTreeNumberPredictionCV,
                           type = "response")
testset$predicted <- round(gbmTest,0)

plotConfusionMatrix(testset,sprintf("Prediction using GBM (%d trees)",optimalTreeNumberPredictionCV))

print(sprintf("Area under curve (AUC) : %.3f",auc(testset$diagnosis, testset$predicted)))

predicted_gbm = testset$predicted

train_matrix = Matrix(as.matrix(trainset %>% select(-diagnosis)), sparse=TRUE)
test_matrix  = Matrix(as.matrix(testset %>% select(-diagnosis,-predicted)), sparse=TRUE)
lightGBM.train = lgb.Dataset(data=train_matrix, label=trainset$diagnosis)
lightGBM.test = lgb.Dataset(data=test_matrix, label=testset$diagnosis)

lightGBM.grid = list(objective = "binary",
                metric = "auc",
                min_sum_hessian_in_leaf = 1,
                feature_fraction = 0.7,
                bagging_fraction = 0.7,
                bagging_freq = 5,
                min_data = 100,
                max_bin = 50,
                lambda_l1 = 8,
                lambda_l2 = 1.3,
                min_data_in_bin=100,
                min_gain_to_split = 10,
                min_data_in_leaf = 30,
                is_unbalance = TRUE)

lightGBM.model.cv = lgb.cv(params = lightGBM.grid, data = lightGBM.train, learning_rate = 0.02, num_leaves = 25,
                   num_threads = 2 , nrounds = 7000, early_stopping_rounds = 50,
                   eval_freq = 20, eval = "auc", nfold = 5, stratified = TRUE)
best.iter = lightGBM.model.cv$best_iter

lightGBM.model = lgb.train(params = lightGBM.grid, data = lightGBM.train, learning_rate = 0.02,
                      num_leaves = 25, num_threads = 2 , nrounds = best.iter,
                      eval_freq = 20, eval = "auc")

#testset$predicted <- round(predict(object = lightGBM.model ,newdata = lightGBM.test),0)

plotConfusionMatrix(testset,"Prediction using lightGBM")


print(sprintf("Area under curve (AUC) : %.3f",auc(testset$diagnosis, testset$predicted)))


dMtrain <- xgb.DMatrix(as.matrix(trainset %>% select(-diagnosis)), label = trainset$diagnosis)
dMtest <- xgb.DMatrix(as.matrix(testset %>% select(-diagnosis,-predicted)), label = testset$diagnosis)

params <- list(
  "objective"           = "binary:logistic",
  "eval_metric"         = "auc",
  "eta"                 = 0.012,
  "subsample"           = 0.8,
  "max_depth"           = 8,
  "colsample_bytree"    =0.9,
  "min_child_weight"    = 5
)

nRounds <- 5000
earlyStoppingRound <- 100
printEveryN = 100
model_xgb.cv <- xgb.cv(params=params,
                      data = dMtrain, 
                      maximize = TRUE,
                      nfold = 5,
                      nrounds = nRounds,
                      nthread = 1,
                      early_stopping_round=earlyStoppingRound,
                      print_every_n=printEveryN)
 
d <- model_xgb.cv$evaluation_log
n <- nrow(d)
v <- model_xgb.cv$best_iteration
df <- data.frame(x=rep(d$iter, 2), val=c(d$train_auc_mean, d$test_auc_mean), 
                   set=rep(c("train", "test"), each=n))
ggplot(data = df, aes(x=x, y=val)) + 
  geom_line(aes(colour=set)) + 
  geom_vline(xintercept=v) + 
  theme_bw() +
  labs(title="AUC values for XGBoost with cross-validation", x="Iteration", y="AUC values (train, test)")
  

model_xgb <- xgboost(params=params,
                      data = dMtrain, 
                      maximize = TRUE,
                      nrounds = nRounds,
                      nthread = 1,
                      early_stopping_round=earlyStoppingRound,
                      print_every_n=printEveryN)
                      
 d <- model_xgb$evaluation_log
n <- nrow(d)
df <- data.frame(x=rep(d$iter), val=d$train_auc)
ggplot(data = df, aes(x=x, y=val)) + 
  geom_line() + 
  theme_bw() +
  labs(title="AUC values for XGBoost", x="Iteration", y="AUC values (train)")
  
  
testset$predicted <- round(predict(object = model_xgb ,newdata = dMtest),0)


plotConfusionMatrix(testset,"Prediction using XGBoost")


print(sprintf("Area under curve (AUC) : %.3f",auc(testset$diagnosis, testset$predicted)))


predicted_xgb = testset$predicted


testset$predicted = 0.3 * predicted_rf + 0.3 * predicted_gbm + 0.4 * predicted_xgb


print(sprintf("Area under curve (AUC) - averaged solution : %.3f",auc(testset$diagnosis, testset$predicted)))
