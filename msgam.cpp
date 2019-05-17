#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() ()
{
  // Read in data 
  DATA_VECTOR(data); 
  DATA_INTEGER(n_states);
  DATA_SPARSE_MATRIX(S); 
  DATA_SPARSE_MATRIX(A); 

  // Read in parameters
  PARAMETER_VECTOR(alpha); 
  PARAMETER_MATRIX(beta); 
  PARAMETER_VECTOR(log_tpm); 
  PARAMETER_VECTOR(log_sigma); 
  PARAMETER_VECTOR(log_lambda); 

  // Unpack and transform parameters
  vector<Type> sigma = exp(log_sigma); 
  // cumulative variance to stop label-switching
  for (int s = 1; s < n_states; ++s) sigma(s) += sigma(s - 1); 
  vector<Type> lambda = exp(log_lambda); 
  
  // Transition probability matrix  
  matrix<Type> tpm(n_states, n_states);
  int cur = 0;
  for (int i = 0; i < n_states; ++i) {
    tpm(i, i) = 1;
   for (int j = 0; j < n_states; ++j) {
    if (i != j) {
      tpm(i, j) = exp(log_tpm(cur)); 
      ++cur;
    }
   }
   tpm.row(i) /= tpm.row(i).sum();
  }

  // Compute stationary distribution
  matrix<Type> delta(1, n_states);
  matrix<Type> I = matrix<Type>::Identity(n_states, n_states);
  matrix<Type> tpminv = I;
  tpminv -= tpm;
  tpminv = (tpminv.array() + 1).matrix();
  matrix<Type> ivec(1, n_states); for (int i = 0; i < n_states; ++i) ivec(0, i) = 1;
  // if tpm is ill-conditioned then just use uniform initial distribution
  try {
    tpminv = tpminv.inverse();
    delta = ivec * tpminv;
  } catch(...) {
    for (int i = 0; i < n_states; ++i) delta(0, i) = 1.0 / n_states;
  }

  // compute spline for each data location
  int n = data.rows();
  matrix<Type> obs(n, n_states);
  for (int s = 0; s < n_states; ++s) obs.col(s) = A * beta.col(s);

  // compute observation probabilities
  matrix<Type> prob(n, n_states);
  matrix<Type> mu(n, n_states); 
  for (int s = 0; s < n_states; ++s) {
    for (int i = 0; i < n; ++i) {
      mu(i, s) = alpha(s) + obs(i, s);
      prob(i, s) = dnorm(data(i), mu(i, s), sigma(s));
    }
  }

  // compute HMM log-likelihood
  Type llk = 0;
  matrix<Type> phi(delta);
  Type sumphi = 0;
  for (int i = 0; i < n; ++i) {
    phi = (phi.array() * prob.row(i).array()).matrix();
    phi = phi * tpm;
    sumphi = phi.sum();
    llk += log(sumphi);
    phi /= sumphi;
  }

  Type nll = -llk;

  // Add smoothing penalties
  density::GMRF_t<Type> spline(S);
  for (int s = 0; s < n_states; ++s) {
    nll -= Type(0.5)*S.cols()*log_lambda(s) - 0.5*lambda(s)*spline.Quadform(beta.col(s));
  }

  // report values back
  ADREPORT(mu); 
  
  return nll;
}
