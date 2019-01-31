# HW 2 

#Q1

library(MASS)
library(glmnet)

N = 500
P = 200

Beta = c(seq(1, 0, length.out = 21), rep(0, P-21))
Beta0 = 0.5

# you must set this seed for generating the data
set.seed(1)

# generate X
V = matrix(0.5, P, P)
diag(V) = 1
X = as.matrix(mvrnorm(N, mu = rep(0, P), Sigma = V))

# generate Y
y = Beta0 + X %*% Beta + rnorm(N)

# check OLS
lm(y ~ X)

# you can write your own code to implement the Lasso
# or you can use the following structure
# in either case, you need to produce a sequence of solutions (of beta_0 and beta_j's) using the X and y defined above.


# now start to write functions to fit the lasso
# prepare the soft thresholding function for updating beta_j (part b)

soft_th <- function(b, lambda)
{
  sign(b) * max(abs(b) - lambda, 0)
}

# initiate lambda as the lambda_max value in part c)
lambda_list=rep(0,P)
for (j in 1:P)
{ 
  sum=0
  for (i in 1:N)
  {
    sum=sum+X[i,j]*(y[i]-mean(y))
  }
  lambda_list[j]=abs(sum/N)
}
lambda_max = max(lambda_list)


# produce a sequence of lambda values 
lambda = exp(seq(log(lambda_max), log(0.01), length.out = 100))

# if you use this formula, you will need to calculate this for the real data too.
LassoFit <- function(X, y, lambda, tol=1e-5, maxiter=50)
{
	# initiate objects to record the values
	mybeta = matrix(NA, ncol(X), length(lambda))
	mybeta0 = rep(NA, length(lambda))
	# mylambda = rep(NA, length(lambda))
	
	# initiate values 
	current_beta = matrix(0, P, 1)
	current_beta0 = mean(y)
	
	for (l in 1:length(lambda))
	{
		# reduce the current lambda value to a smaller one
		current_lambda = lambda[l]
		
		for (k in 1:maxiter)
		{
		  current_beta_last=current_beta
			# update the intercept term based on the current beta values. 
		  current_beta0 = sum(y - X %*% current_beta)/N
			
			# compute residuals (this is with all variables presented)
			r = y - current_beta0 - X %*% current_beta

			# start to update each beta_j 
			
			for (j in 1:ncol(X))
			{
				# remove the effect of variable j from model, and compute the residual
				r = r+X[,j]*current_beta[j]
				
				# update beta_j using the results in part b)
        current_beta[j]=soft_th(X[,j]%*%r,current_lambda*N)/(t(X[,j]%*%X[,j]))
				
				
				# add the effect of variable j back to the model, and compute the residual
				r = r-X[,j]*current_beta[j]
			}
			
			# check if beta changed more than the tolerance level in this iteration (use tol as the threshold)
			# if not, break out of this loop k
			# you will need to record the beta values at the start of this iteration for comparison. 
			if ((sum(abs(current_beta-current_beta_last))<tol)&&k>1) break;
		}
		
		# record the beta_j and beta_0 values
		mybeta[, l] = current_beta
		mybeta0[l] = current_beta0
	}
	
	return(list("beta" = mybeta, "b0" = mybeta0, "lambda" = lambda))
}

# now, perform the Lasso mode on the simulated dataset 

#test data
X_test = as.matrix(mvrnorm(1000, mu = rep(0, P), Sigma = V))
Y_test = Beta0 + X_test %*% Beta + rnorm(N)

fit=LassoFit(X, y, lambda, tol=1e-5, maxiter=100)
Y_pred=X_test%*%fit$beta+matrix(1, 1000, 1)%*%t(fit$b0)
error=Y_test%*%matrix(1, 1, 100)-Y_pred
error_list=rep(0,100)
for (i in 1:100)
{
  error_list[i]=sum(error[,i]^2)
}
#lambda that has the minimum prediction error
fit$lambda[which.min(error_list)]



#Q2

#a
cv=cv.glmnet(X, y, nfolds=10)
model=glmnet(X,y,lambda=cv$lambda.min)
coefficient=coef(model)
#best lambda
print(cv$lambda.min)
#number of nonzeros parameters
length(which(coefficient!= 0))



#b
#estimate degree of freedom
Y=matrix(0,500,20)
Y_pred=matrix(0,500,20)
for (i in 1:20)
{
  Y[,i]= Beta0 + X %*% Beta + rnorm(N)
  model=glmnet(X,Y[,i],lambda=cv$lambda.min)
  Y_pred[,i]=predict(model,X)
}

df_estimate=0
for (i in 1:500)
{
  df_estimate=df_estimate+cov(Y[i,],Y_pred[i,])
}
print(df_estimate)


#c
fit=lm.ridge(y ~ X, lambda = seq(0,100,0.001))
best_lambda = fit$lambda[(which.min(fit$GCV))]
model=lm.ridge(y ~ X, lambda=best_lambda)
coefficient=coef(model)
#best lambda
print(best_lambda)
#number of nonzeros parameters
length(which(coefficient!= 0))

#theoratical value
df_theory=sum(diag(X%*%solve(t(X)%*%X+best_lambda*diag(200))%*%t(X)))
print(df_theory)

#estimated value
Y=matrix(0,500,20)
Y_pred=matrix(0,500,20)
for (i in 1:20)
{
  Y[,i]= Beta0 + X %*% Beta + rnorm(N)
  model=lm.ridge(Y[,i] ~ X, lambda =best_lambda)
  Y_pred[,i]=cbind(const=1, X) %*% coef(model)
}

df_estimate=0
for (i in 1:500)
{
  df_estimate=df_estimate+cov(Y[i,],Y_pred[i,])
}
print(df_estimate)



