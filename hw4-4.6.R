# Handwritten digit classification
#a)
# trn3, tst3: all label = 3;  trn5, tst5: all label = 5
trn3 = read.table("/Users/weiweili/Documents/UCSD-Stats/CSE 250A/hw4 data/newTrain3.txt")
tst3 = read.table("/Users/weiweili/Documents/UCSD-Stats/CSE 250A/hw4 data/newTest3.txt")
trn5 = read.table("/Users/weiweili/Documents/UCSD-Stats/CSE 250A/hw4 data/newTrain5.txt")
tst5 = read.table("/Users/weiweili/Documents/UCSD-Stats/CSE 250A/hw4 data/newTest5.txt")


test = rbind(as.matrix(tst3), as.matrix(tst5))

train = rbind(as.matrix(trn3), as.matrix(trn5))

#log-likelihood
L = function(w,x,y){
  # x predictor matrix: m*n
  # y binary: m*1 vector 
  # w weight: n*1 vector
  # L first colum is loglikelihood, the rest columns are gradients
  q = 1/(1+exp( -x%*%w ))   # dim m*1
  l= y*log(q)+(1-y)*log(1-q) #dim m*1  
  return(sum(l)) 
}

#gradient
g = function(w,x,y){
  # x predictor matrix: m*n
  # y binary: m*1 vector 
  # w weight: n*1 vector
  # g gradients: n*1
  q = 1/(1+exp( -x%*%w ))   # dim m*1
  g= diag(as.vector(y-q))%*%x  #dim m*n
  return( colSums(g) )
}


#hessian
hessian = function(w,x){
  # w weight: n*1 vector
  # x predictor matrix: m*n
  q = 1/( 1+exp(-x%*%w) )   # dim m*1
  c = q*(1-q) # dim m*1
  n = ncol(x)
  H =  matrix(NA,n,n)
  for (i in 1:n){
    for (j in 1:i){
      H[i,j] = -sum(c*x[,i]*x[,j])
      H[j,i] = H[i,j]
    }
  }
  return(H)
}


error = function(w,x,y) {
  # x predictor matrix: m*n
  # y binary: m*1 vector 
  # w weight: n*1 vector
  # L first colum is loglikelihood, the rest columns are gradients
  q = 1/(1+exp( -x%*%w ))   # dim m*1
  #l = abs(y-q)   #error
  l=  (y - q)^2 #dim m*1  #error
  return(sum(l)/nrow(x))   
}
  




y1 = matrix(0,nrow(trn3), 1)
y2 = matrix(1,nrow(trn5), 1)
y =  rbind(y1, y2)
test.y = matrix(0,nrow(tst3),1)
test.y = rbind(test.y, matrix(1,nrow(tst5), 1))



#gradient ascent:
w=matrix(rnorm(64,0,1),nrow=64, ncol=1)
k = 0.02/nrow(train)

lh2 = L(w, train, y)
dif = 1
N = 100
LH = matrix(nrow = N, ncol = 1)
err = matrix(nrow = N, ncol = 1)
i = 1
while(dif >= 0 & i<=N){
  lh = lh2
  LH[i] = lh
  err[i] = error(w,test,test.y)
  oldw = w
  grad = g(w, train, y)
  w = w+k*(grad)
  lh2 = L(w, train, y)
  dif = lh2 - lh
  i=i+1
}

if(dif < 0){w = oldw}
plot(seq(1,i-1),LH[1:i-1], type = "l", col = "red", xlab = "number of iterations", ylab = "log-likelihood")
plot(seq(1,i-1),err[1:i-1], type = "l", col = "blue", xlab = "number of iterations", ylab = "error rate")






#Newton:
norm_vec <- function(x) sqrt(sum(x^2))

w=matrix(0,nrow=64, ncol=1)

lh2 = L(w, train, y)
dif = 1
N = 40
LH = matrix(nrow = N, ncol = 1)
err = matrix(nrow = N, ncol = 1)
i = 1
while( dif > 0 & i<=N){
  lh = lh2
  LH[i] = lh
  err[i] = error(w,test,test.y)
  oldw = w
  grad = g(w, train, y)
  H = hessian(w,train)
  if ( det(H) < 1e-6 ) {
    aH = H + 0.001*diag(1,ncol(train),ncol(train))  
    k2 = solve(aH,grad)
  } else {
    k2 = solve(H,grad)
  }
  w = w-k2
  lh2 = L(w, train, y)
  #grad2 = g(w,train,y)
  #dif = norm_vec(grad2 - grad)
  dif = lh2 - lh
  i=i+1
}


if(dif < 0){w = oldw}
plot(seq(1,i-1),LH[1:i-1], type = "l", col = "red", xlab = "number of iterations", ylab = "log-likelihood")
plot(seq(1,i-1),err[1:i-1], type = "l", col = "blue", xlab = "number of iterations", ylab = "error rate")

