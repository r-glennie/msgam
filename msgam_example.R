library(MSwM) 
library(TMB) 

compile("msgam.cpp", flags = "-Wno-ignored-attributes")
dyn.load(dynlib("msgam"))

# simulate data 
set.seed(25821)
n <- 1000
true_tpm <- matrix(c(0.8, 0.3, 0.2, 0.7), nr = 2, nc = 2) 
f <- function(x, s) {
  if (s == 1) return(cos(5*x))
  if (s == 2) return(sin(5*x)*x)
} 
intercept <- c(1.5, 2.0) 
true_sigma <- c(0.2, 0.4)
y <- rep(0, n) 
s <- rep(0, n + 1) 
s[1] <- 1 
x <- seq(0, 1, length = n)
for (i in 1:n) {
  y[i] <- rnorm(1, intercept[s[i]] + f(x[i], s[i]), true_sigma[s[i]])
  s[i+1] <- sample(1:2, prob = true_tpm[s[i],], size = 1) 
}
s <- s[-(n+1)] 

# plot it 
col <- ifelse(s == 1, "blue", "red") 
plot(x, y, col = col, pch = 20)

# compute smoothing matrices in mgcv 
library(mgcv) 
gam <- gam(y ~ s(x, bs = "bs"), fit = FALSE) 
S <- as(gam$S[[1]], "sparseMatrix") 

# set number of states
n_states <- 2 

# set up parameters 
pars <- list(alpha = rep(mean(y), n_states), 
	     beta = matrix(0, nr = ncol(S), nc = n_states), 
             log_tpm = rep(0, n_states^2 - n_states), 
             log_sigma = rep(log(sd(y)), n_states), 
             log_lambda = rep(0, n_states)) 

# set up data 
dat <- list(data = y, 
            n_states = n_states, 
	    S = S, 
            A = as(gam$X[,-1], "sparseMatrix")) 

# create model object
obj <- MakeADFun(dat, pars, random = "beta", dll = "msgam") 

# fit model
fit <- nlminb(start = obj$par, objective = obj$fn, gradient = obj$gr) 

# get results
res <- sdreport(obj)
mu <- matrix(res$value, nr = n, nc = 2)
sd <- matrix(res$sd, nr = n, nc = 2) 
lcl <- mu - qnorm(0.975) * sd 
ucl <- mu + qnorm(0.975) * sd 

# plot it 
plot(x, y, pch = 20, col = "grey80")
lines(x, mu[,1], col = "blue", lwd = 2) 
lines(x, mu[,2], col = "red", lwd = 2)
lines(x, lcl[,1], col = "blue", lwd = 2, lty = 2) 
lines(x, ucl[,1], col = "blue", lwd = 2, lty = 2) 
lines(x, lcl[,2], col = "red", lwd = 2, lty = 2) 
lines(x, ucl[,2], col = "red", lwd = 2, lty = 2) 

# estimated tpm 
tpm <- matrix(0, nr = 2, nc = 2) 
tpm[!diag(2)] <- exp(fit$par[4:3]) 
diag(tpm) <- 1 
tpm <- tpm / rowSums(tpm)

# estimated variances 
sigma <- cumsum(exp(fit$par[5:6]))





