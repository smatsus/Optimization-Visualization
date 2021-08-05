import numpy as np
import time
import sys
import math
import copy
import itertools 
import random
import pprint
random.seed(1)
def includes(phi,x):
  return set(phi).issubset(set(x))

class LLM:
  def __init__ (self, B, S):

    self.B_ = B # list of tuples
    self.S_ = S # list of tuples
    self.set_Bsub(self.S_, self.B_) #dict (key: tuple, val: list of tuples)
    self.set_Ssub(self.S_, self.B_)  # dict (key: tuple, val: list of tuples)
    self.init_theta() # dict (key: list, val: float)
    self.P_ = {}
    self.Phat_ = {}  # dict (key: list, val: float)
    self.etahat_ = {}  # dict (key: list, val: float)
    self.parameter_ = []
    self.param_value_list = []
    
    
    
    
  def fit(self, X, n_iter, stepsize=-1, mu = 0, lambda_ = -1, solver="grad",mode = "random", alpha = 0):
    """Actual implementation of LLM on posets fitting.
        Parameters
        ----------
        X : array-like, shape (n_samples,)
            Training vectors, where n_samples is the number of samples and 
            each element is a tuple of indices of 1s.
        n_iter: number of iteration
        method: "grad"  = gradient descent, "coor" coordinate descent
        Returns
        -------
        self : object
    """
    self.compute_Phat(X)
    self.compute_etahat(X)
    if solver == "grad":
      self.gradient_descent(X, n_iter, stepsize)
    elif solver == "coor":
      self.coordinate_descent(X, n_iter)
    elif solver == "acc_grad":
      self.accelerated_gradient_descent(X, n_iter, stepsize, mu)
    elif solver == "ACDM":
      self.ACDM(X, n_iter)
    elif solver == "mix_exact_ACDM":
      self.mix_exact_ACDM(X, n_iter)
    elif solver == "efficient_ACDM":
      self.efficient_ACDM(X, n_iter)
    elif solver == "efficient_mix_exact_ACDM":
      self.efficient_mix_exact_ACDM(X, n_iter)
    elif solver == "efficient_full_exact_ACDM":
      self.efficient_full_exact_ACDM(X, n_iter)
    elif solver == "check_two_dimesion_converge":
      self.check_two_dimesion_converge(X, n_iter)
    elif solver == "check_two_dimesion_converge_points":
      self.check_two_dimesion_converge_points(X, n_iter)
    else:
      print("Solver Option Error", file = sys.stderr)


    return self 
  

  def set_Bsub(self, S, B):
    """
    Phi_[x]= { phi in B | x includes phi }
    """
    self.Bsub_ = {():[]}

    for x in S:
      B_x = []
      for phi in B:
        if includes(phi,x):
          B_x.append(phi)
      self.Bsub_[x] = B_x



  def set_Ssub(self, S, B):
    """
    Ssub [phi]= { x in S | x includes phi } 
    """
    self.Ssub_ = {}
    for phi in B:
      ssub = []
      for x in S:
        if includes(phi,x):
          ssub.append(x)
      self.Ssub_[phi] = ssub

  def compute_Phat(self, X):
    for x in self.S_:
      self.Phat_[x] = 0.0
    for xi in X:
      self.Phat_[xi] += 1 / len(X)

  def compute_etahat(self, X):
    """        
        Parameters
        ----------
        X : array-like, shape (n_samples,)
            Training vectors, where n_samples is the number of samples and 
            each element is a tuple of indices of 1s.
            
    """
    for phi in self.B_:
      etahat_phi = 0.0
      for xi in self.Ssub_[phi]:
        etahat_phi += self.Phat_[xi] 
      self.etahat_[phi] = etahat_phi


  def compute_logP(self, x):
    """
    Computing lopP
    Parameters
    -----
    xi : i th row vector of X
    
    Returns
    -----
    ret : logP
    
    """
    ret = 0.0
    for phi in self.Bsub_[x]:
      if phi != ():
        ret +=  self.theta_[phi]
    ret += self.theta_perp
    return ret 


  def init_theta(self):    
    self.theta_ = {}
    for phi in self.S_:
      if phi != ():
        self.theta_[phi] = 0.0
    self.compute_theta_perp()   

  def compute_theta_perp(self):
    """
    Computing self.theta_perp     
    """

    r = 0.0 
    for x in self.S_:
      s = 0.0 # sum theta_phi phi(x) for theta in B
      for phi in self.Bsub_[x]:
        if phi != ():
          s += self.theta_[phi]
      r += np.e ** (s)
    self.theta_perp = -np.log(r)
  
  def compute_P(self):
    """
    Computing P
    Returns
    -----
    P_ : len(P_) = len(S_)
    """
    for x in self.S_:
      self.P_[x] = np.e ** ( self.compute_logP(x) )

  def compute_eta(self):
    """
    Computing eta
    Returns
    -----
    eta_ : dict type
             len(eta_) = len(B_)
    """
    self.eta_ = {}
    for phi in self.B_ :
      self.eta_[phi] = 0.0
      for x in self.Ssub_[phi] :
        self.eta_[phi] += self.P_[x]

  def compute_KL(self):
    """
    Computing KL_divergence sum_{x in S} Phat(x) ( ln Phat(x)  - ln P(x) )
    
    """
    ret = 0.0
    for x in self.S_:
      Phatx = self.Phat_[x]
      if Phatx != 0.0:
        ret += Phatx * np.log(Phatx / self.P_[x] )

    return ret
  

  def gradient_descent(self, X, n_iter, step):  
    """
    Actual implementation gradient_descent
    Parameters 
    -----
    X : array-like, shape (n_samples,)
            Training vectors, where n_samples is the number of samples and 
            each element is a tuple of indices of 1s. 
    n_iter
    step 
    """

    start = time.time()
    for iter in range(n_iter):
      self.compute_P()
      self.compute_eta()
      kl=self.compute_KL()
      print(iter ,":", "KL divergence: ", f'{kl:.16f}' ," time : %4.2f"% (time.time()-start),flush=True)
        
      for phi in self.B_:
        new_theta_phi = self.theta_[phi] + step * (self.etahat_[phi] - self.eta_[phi] )
        self.theta_[phi] = new_theta_phi

      self.compute_theta_perp()

  def update_accelerated_theta(self, iter, step, mu):      
    
    if iter == 0:

      self.grad_list = [] 
      # [theta(t) - step*grad( L_D( theta(t)) ), theta_(t-1) - step* ( L_D(theta(t-1)) ) ]
      self.grad_theta_ = {}
      # theta^(t-1) - step * grad( L_D( theta^(t-1) ) )
      self.lambda_list = [0,1]
      self.lambda_ = 1
      for phi in self.B_:
        grad_new_theta_phi = self.theta_[phi] + step * (self.etahat_[phi] - self.eta_[phi] )
        self.grad_theta_[phi] = grad_new_theta_phi
        self.theta_[phi] = grad_new_theta_phi
      self.grad_list.append(self.grad_theta_)
      
    elif iter == 1:
      pre_grad_theta_ = copy.copy(self.grad_theta_)
      pre_lambda_ = copy.copy(self.lambda_)
      self.lambda_ = (1 + math.sqrt( 1+ 4 * (pre_lambda_**2)))/2
      for phi in self.B_:
        grad_new_theta_phi = self.theta_[phi] + step * (self.etahat_[phi] - self.eta_[phi] )
        self.grad_theta_[phi] = grad_new_theta_phi
        if mu == 0:
          new_theta_phi = grad_new_theta_phi   #no hyperparameter
        else:
          new_theta_phi = (1 + mu) * grad_new_theta_phi - mu * pre_grad_theta_[phi]

        self.theta_[phi] = new_theta_phi
      self.grad_list = [pre_grad_theta_,self.grad_theta_]
      self.lambda_list = [pre_lambda_,self.lambda_]
    else:
      pre_grad_theta_ = copy.copy(self.grad_theta_)
      pre_lambda_ = copy.copy(self.lambda_)
      self.lambda_ = (1 + math.sqrt(1 + 4 * (pre_lambda_**2)))/2
      self.gamma = (1 - pre_lambda_) / self.lambda_

      for phi in self.B_:
        grad_new_theta_phi = self.theta_[phi] + step * (self.etahat_[phi] - self.eta_[phi])
        self.grad_theta_[phi] = grad_new_theta_phi
        if mu ==0:
          new_theta_phi = (1 -self.gamma) * grad_new_theta_phi + self.gamma * pre_grad_theta_[phi]      #no hyperparameter
        else:
          new_theta_phi = (1 + mu) * grad_new_theta_phi - mu * pre_grad_theta_[phi]

        self.theta_[phi] = new_theta_phi
        
      self.grad_list = [pre_grad_theta_,self.grad_theta_]
      self.lambda_list = [pre_lambda_,self.lambda_]

  def accelerated_gradient_descent(self, X, n_iter, step, mu):
    """                                                                                                     
    Actual implementation accelerated_gradient_descent                                                          
    Parameters                                                                                             
    -----                                                                                                   
    X : array-like, shape (n_samples,)                                                                       
            Training vectors, where n_samples is the number of samples and                                   
            each element is a tuple of indices of 1s.                                                       
    n_iter                                                                                                   
    step                                                                                                      
    mu : momentum
    """

    start = time.time()
    for iter in range(n_iter):
      self.compute_P()
      self.compute_eta()
      kl=self.compute_KL()
      print(iter ,":", "KL divergence: ", f'{kl:.16f}' ," time : %4.2f"% (time.time()-start),flush=True)
      self.update_accelerated_theta(iter, step ,mu)

      self.compute_theta_perp()
      
  def coordinate_descent(self, X, max_epoch):
    """
    Actual implementation coodinate_descent
    Parameters 
    -----
    X : array-like, shape (n_samples,)
            Training vectors, where n_samples is the number of samples and 
            each element is a tuple of indices of 1s.
    max_epoch
    -----
    
    """
    u = {}
    for x in self.S_:
      u[x] = 1.0

    Z = len(self.S_)
    """
    kl = 0.0
    for x in self.S_:
      Phatx = self.Phat_[x]
      if Phatx != 0.0:
        kl += Phatx * np.log(Phatx)
    kl += np.log(Z)
    """
    start = time.time()
    
    for epoch in range(max_epoch):
      
      #compute KL
      kl = 0.0
      for x in self.S_:
        if self.Phat_[x] != 0.0:
          kl += self.Phat_[x] * np.log(self.Phat_[x] / u[x])
      kl += np.log(Z)
      print(epoch ,":", "KL divergence: ", f'{kl:.16f}' ," time : %4.2f"% (time.time()-start),flush=True)
      
      #print(epoch ,":",  "KL divergence:",f'{kl:.16f}' ," time : %4.2f"% (time.time()-start), "exactZ - Z :", sum(u.values())-Z,  flush=True)
      #random.shuffle(self.B_)
      for phi in self.B_:
        etahat_phi = self.etahat_[phi] 

        if etahat_phi >= 1.0 or etahat_phi == 0.0:
          
          continue

        #compute eta_phi
        eta_phi = 0.0
        for x in self.Ssub_[phi]: 
            eta_phi += u[x]
        eta_phi /= Z
        
        #compute exp(delta)
        exp_delta = 1 + (etahat_phi - eta_phi) / eta_phi / (1 - etahat_phi)
        delta = np.log1p( (etahat_phi - eta_phi) / eta_phi / (1 - etahat_phi) )
        


        #update theta_phi
        #self.theta_[self.invB_[phi]] += np.log(exp_delta)
        self.theta_[phi] += delta

        #update u,Z
        #pre_Z = Z
        for x in self.Ssub_[phi]:
          #diff_ux = u[x] * np.expm1(delta)
          #u[x] += diff_ux
          #Z += diff_ux

          Z += u[x]* (etahat_phi - eta_phi) / eta_phi / (1 - etahat_phi)  #u[x]*(exp_delta - 1)
          u[x] *= exp_delta
          #kl -= self.Phat_[x] * delta
        #Z = sum(u.values())
        #kl += np.log(Z / pre_Z)
    #print(kl)
    self.theta_perp = -np.log(Z)
    for x in self.S_:
      self.P_[x] = u[x]/Z
    
    #print('KL(P^||P) : '+ str(self.compute_KL()))



  def accelerated_coordinate_descent(self, max_epoch):
    """                                                                                    
    Actual implementation coodinate_descent                                                
    Parameters                                                                             
    -----                                                                                  
    X : array-like, shape (n_samples,)                                                     
 
            Training vectors, where n_samples is the number of samples and                 
 
            each element is a tuple of indices of 1s.                                      
    max_epoch                                                                              
    -----
    """
    u = {}
    for x in self.S_:
      u[x] = 1.0

    Z = len(self.S_)
    self.alphas = [0.25]
    self.ys = np.zeros(len(self.B_))
    self.thetas = [np.zeros(len(self.B_))]

    start = time.time()
    for epoch in range(max_epoch):

      #compute KL
      kl = 0.0
      for x in self.S_:
        Phatx = self.Phat_[x]
        if Phatx != 0.0:
          kl += Phatx * (np.log(Phatx) - np.log(u[x]/Z))

      print(epoch ,":",  "KL divergence:",f'{kl:.16f}' ," time : %4.2f"% (time.time()-start),  flush=True)

      #index = np.random.RandomState().permutation(range(len(self.B_))) #permutative        
 
      #index = np.random.randint(0,len(self.B_)-1,len(self.B_))  #random                   


      for iter in range(len(self.B_)):
        phi = self.B_[ index[iter] ]
        etahat_phi = self.etahat_[phi]

        if etahat_phi >= 1.0 or etahat_phi == 0.0:
          continue
        
        #self.update_parameters(epoch,iter)

        #compute eta_phi                                                                   
        
        eta_phi = 0.0
        for x in self.Ssub_[phi]:
            eta_phi += u[x]
        eta_phi /= Z

        #compute exp(delta)                                                                
 
        exp_delta = (1-eta_phi)/eta_phi * etahat_phi/(1-etahat_phi)

        #update theta_phi                                                                  
 
        self.theta_[phi] += np.log(exp_delta)                                 
        #update Z
        for x in self.Ssub_[phi]:
          Z += u[x] * (exp_delta - 1)

        self.theta_perp = - np.log(Z)                                                     

        #update u                                                                          
        for x in self.Ssub_[phi]:
          u[x] *= exp_delta
  
  def ACDM(self, X, n_iter):
    calc_KL_time = 0
    self.theta_tilde = dict()
    self.theta_hat = dict()
    B_size = len(self.B_)
    for phi in self.B_:
      self.theta_tilde[phi] = 0.0
    gamma = 1/(2.0*B_size)
    start = time.time()

    for iter_ in range(n_iter):
      for num_ in range(B_size):
        phi = random.choice(self.B_)
        if self.etahat_[phi] >= 1.0 or self.etahat_[phi] == 0.0:
          continue
        # calculate Coefficients
        previous_gamma = gamma
        gamma = (1+math.sqrt(1+4*B_size**2*previous_gamma**2))/(2*B_size)
        alpha = 1/(B_size*gamma)
        # Update theta_hat
        for phi_ in self.B_:
          self.theta_hat[phi_] = alpha*self.theta_tilde[phi_] + (1-alpha)*self.theta_[phi_]
        # calculate naively eta_theta_hat
        self.compute_theta_perp_hat()
        self.compute_P_theta_hat()
        self.compute_eta_theta_hat(phi)
        # update theta
        new_theta_phi = self.theta_hat[phi] + 4 * (self.etahat_[phi] - self.eta_theta_hat[phi] )
        self.theta_[phi] = new_theta_phi
        for phi_ in self.B_:
          if phi_ != phi:
            self.theta_[phi_] = self.theta_hat[phi_]
        # update theta_tilde
        self.theta_tilde[phi] = self.theta_tilde[phi] + 4*gamma*(self.etahat_[phi] - self.eta_theta_hat[phi])

      temp_time = time.time()
      self.compute_theta_perp()
      self.compute_P()
      kl=self.compute_KL()
      end_time = time.time() - temp_time
      calc_KL_time += end_time 
      print(iter_ ,":", "KL divergence: ", f'{kl:.16f}' ," time : %4.2f"% (time.time()-calc_KL_time-start),flush=True)
 
  def compute_eta_theta_hat(self, phi):
    """
    Computing eta
    Returns
    -----
    eta_ : dict type
              len(eta_) = len(B_)
    """
    self.eta_theta_hat = {}
    self.eta_theta_hat[phi] = 0.0
    for x in self.Ssub_[phi]:
      self.eta_theta_hat[phi] += self.P_theta_hat[x]

  def compute_P_theta_hat(self):
    """
    Computing P
    Returns
    -----
    P_ : len(P_) = len(S_)
    """
    self.P_theta_hat = dict()
    for x in self.S_:
      self.P_theta_hat[x] = np.e ** (self.compute_logP_theta_hat(x) )

  def compute_logP_theta_hat(self, x):
    """
    Computing lopP
    Parameters
    -----
    xi : i th row vector of X
    
    Returns
    -----
    ret : logP
    
    """
    ret = 0.0
    for phi in self.Bsub_[x]:
      if phi != ():
        ret +=  self.theta_hat[phi]
    ret += self.theta_perp_hat
    return ret 

  def compute_theta_perp_hat(self):
    """
    Computing self.theta_perp
    """

    r = 0.0 
    for x in self.S_:
      s = 0.0 # sum theta_phi phi(x) for theta in B
      for phi in self.Bsub_[x]:
        if phi != ():
          s += self.theta_hat[phi]
      r += np.e ** (s)
    self.theta_perp_hat = -np.log(r)

  def compute_KL_hat(self):
    """
    Computing KL_divergence sum_{x in S} Phat(x) ( ln Phat(x)  - ln P(x) )
    """
    ret = 0.0
    for x in self.S_:
      Phatx = self.Phat_[x]
      if Phatx != 0.0:
        ret += Phatx * np.log(Phatx / self.P_theta_hat[x] )

    return ret


  def mix_exact_ACDM(self, X, n_iter):
    calc_KL_time = 0
    self.theta_tilde = dict()
    self.theta_hat = dict()
    B_size = len(self.B_)
    for phi in self.B_:
      self.theta_tilde[phi] = 0.0
    gamma = 1/2.0
    start = time.time()

    for iter_ in range(n_iter):
      for num_ in range(B_size):
        phi = random.choice(self.B_)
        if self.etahat_[phi] >= 1.0 or self.etahat_[phi] == 0.0:
          continue
        # calculate Coefficients
        previous_gamma = gamma
        gamma = (1+math.sqrt(1+4*previous_gamma**2))/2.0
        alpha = 1/gamma
        # Update theta_hat
        for phi_ in self.B_:
          self.theta_hat[phi_] = alpha*self.theta_tilde[phi_] + (1-alpha)*self.theta_[phi_]
        # calculate naively eta_theta_hat
        self.compute_theta_perp_hat()
        self.compute_P_theta_hat()
        self.compute_eta_theta_hat(phi)
        #delta
        delta = np.log((1.0 -self.eta_theta_hat[phi])* self.etahat_[phi]/(self.eta_theta_hat[phi]*(1-self.etahat_[phi])))

        # update theta
        new_theta_phi = self.theta_hat[phi] + delta
        self.theta_[phi] = new_theta_phi
        for phi_ in self.B_:
          if phi_ != phi:
            self.theta_[phi_] = self.theta_hat[phi_]
        # update theta_tilde
        self.theta_tilde[phi] = self.theta_tilde[phi] + 4/B_size*gamma*(self.etahat_[phi] - self.eta_theta_hat[phi])

      temp_time = time.time()
      self.compute_theta_perp()
      self.compute_P()
      kl=self.compute_KL()
      end_time = time.time() - temp_time
      calc_KL_time += end_time 
      print(iter_ ,":", "KL divergence: ", f'{kl:.16f}' ," time : %4.2f"% (time.time()-calc_KL_time-start),flush=True)


  def efficient_mix_exact_ACDM(self, X, n_iter):
    """
    The implementation of efficient stable ACDM.
    Parameters 
    -----
    X : array-like, shape (n_samples,)
            Training vectors, where n_samples is the number of samples and 
            each element is a tuple of indices of 1s. 
    n_iter
    step 
    """
    calc_KL_time = 0
    e = np.e 
    B_size = len(self.B_)
    Z = len(self.S_)
    total_step = n_iter*B_size
    # Initialization of Theta
    theta_prime_tilde = dict()
    theta_prime_hat = dict()
    for phi in self.B_:
      theta_prime_tilde[phi] = 0.0
      theta_prime_hat[phi] = 0.0

    # Initialization of U
    U_hat = dict() 
    U_prime_hat = dict() 
    U_prime_tilde = dict()
    for x in self.S_:
      U_hat[x] = 0
      U_prime_hat[x] = 0.0
      U_prime_tilde[x] = 0.0
    # Initialization of Coefficients
    alpha = [0] * (total_step+1)
    gamma = [0] * (total_step+2)
    c_hat = [0] * (total_step+2)
    gamma[-1] =1/(2.0) 
    c_hat[-1] = 3.0 + 2*math.sqrt(2.000)    
    for k in range(total_step+1):
      gamma_k_before = gamma[k-1]
      gamma_k = math.sqrt(1.0/(4.0)+gamma_k_before**2)+1.0/(2.0)
      gamma[k] = gamma_k
      alpha[k] =1/(gamma_k)
      c_hat[k] =  (1-alpha[k]) * c_hat[k-1]
    # Caching 
    e_U_hat_dict = dict() 
    for x in self.S_:
      e_U_hat_dict[x] = 1.0
    # Beginning of Iteration
    start = time.time()
    for iter_ in range(n_iter):
      for index in range(B_size):
        k = iter_*B_size + index
        phi = random.choice(self.B_)
        if self.etahat_[phi] >= 1.0 or self.etahat_[phi]== 0.0:
          continue
        # Caching
        c_hat_k = c_hat[k]
        c_hat_k_after = c_hat[k+1]
        alpha_k_after = alpha[k+1]
        gamma_k = gamma[k]

        # calculate G, Z
        G = 0 
        for x in self.Ssub_[phi]:
          G += e_U_hat_dict[x]
        eta_ = G/Z
        # update tilde_theta_prime, hat_theta_prime
        nabla = 4.0*(self.etahat_[phi] - eta_)
        nabla_exact = np.log((1.0 -eta_)* self.etahat_[phi]/(eta_*(1-self.etahat_[phi])))

        delta_tilde = nabla
        delta_hat = nabla_exact/len(X)
        step1 = gamma_k/B_size * delta_tilde
        step2 = 1.0/c_hat_k_after*((alpha_k_after-(1-c_hat_k_after))*gamma_k/B_size*delta_tilde+(1-alpha_k_after)*delta_hat)


        theta_prime_tilde[phi] += step1
        theta_prime_hat[phi] += step2 
        for x in self.Ssub_[phi]:
          U_prime_tilde[x] += step1
          U_prime_hat[x] += step2
          U_hat_temp = (1-c_hat_k_after)*U_prime_tilde[x]+c_hat_k_after*U_prime_hat[x]
          e_U_hat_temp = e ** U_hat_temp
          Z += e_U_hat_temp - e_U_hat_dict[x]
          e_U_hat_dict[x] = e_U_hat_temp

      update_start_time = time.time()
      for x in self.B_:
        self.theta_[x] = (1-c_hat_k_after)*theta_prime_tilde[x]+c_hat_k_after*theta_prime_hat[x]
      self.theta_[phi] += 4*(self.etahat_[phi]-eta_)
      self.compute_theta_perp()
      self.compute_P()
      kl=self.compute_KL()
      calc_KL_time +=  time.time() - update_start_time 
      print(iter_ ,":", "KL divergence: ", f'{kl:.16f}' ," time : %4.2f"% (time.time()-calc_KL_time-start),flush=True)

  def efficient_ACDM(self, X, n_iter):
    """
    The implementation of efficient stable ACDM.
    Parameters 
    -----
    X : array-like, shape (n_samples,)
            Training vectors, where n_samples is the number of samples and 
            each element is a tuple of indices of 1s. 
    n_iter
    step 
    """
    calc_KL_time = 0
    e = np.e 
    B_size = len(self.B_)
    Z = len(self.S_)
    total_step = n_iter*B_size
    # Initialization of Theta
    theta_prime_tilde = dict()
    theta_prime_hat = dict()
    for phi in self.B_:
      theta_prime_tilde[phi] = 0.0
      theta_prime_hat[phi] = 0.0

    # Initialization of U
    U_hat = dict() 
    U_prime_hat = dict() 
    U_prime_tilde = dict()
    for x in self.S_:
      U_hat[x] = 0
      U_prime_hat[x] = 0.0
      U_prime_tilde[x] = 0.0
    # Initialization of Coefficients
    alpha = [0] * (total_step+1)
    gamma = [0] * (total_step+2)
    c_hat = [0] * (total_step+2)
    gamma[-1] =1/(2.0) 
    c_hat[-1] = 3.0 + 2*math.sqrt(2.000)    
    for k in range(total_step+1):
      gamma_k_before = gamma[k-1]
      gamma_k = math.sqrt(1.0/(4.0)+gamma_k_before**2)+1.0/(2.0)
      gamma[k] = gamma_k
      alpha[k] =1/(gamma_k )
      c_hat[k] =  (1-alpha[k]) * c_hat[k-1]
    # Caching 
    e_U_hat_dict = dict() 
    for x in self.S_:
      e_U_hat_dict[x] = 1.0
    # Beginning of Iteration
    start = time.time()
    for iter_ in range(n_iter):
      for index in range(B_size):
        k = iter_*B_size + index
        phi = random.choice(self.B_)
        # Caching
        c_hat_k = c_hat[k]
        c_hat_k_after = c_hat[k+1]
        alpha_k_after = alpha[k+1]
        gamma_k = gamma[k]

        # calculate G, Z
        G = 0 
        for x in self.Ssub_[phi]:
          G += e_U_hat_dict[x]
        eta_ = G/Z
        # update tilde_theta_prime, hat_theta_prime
        nabla = (self.etahat_[phi] - eta_)
        step1 = 4.0 * gamma_k/B_size
        step2 = 4.0/c_hat_k_after*(1-alpha_k_after-(alpha_k_after+1-c_hat_k_after)*gamma_k/B_size)

        theta_prime_tilde[phi] += step1 * nabla
        theta_prime_hat[phi] += step2 * nabla
        for x in self.Ssub_[phi]:
          U_prime_tilde[x] += step1 * nabla
          U_prime_hat[x] += step2 * nabla
          U_hat_temp = (1-c_hat_k_after)*U_prime_tilde[x]+c_hat_k_after*U_prime_hat[x]
          e_U_hat_temp = e ** U_hat_temp
          Z += e_U_hat_temp - e_U_hat_dict[x]
          e_U_hat_dict[x] = e_U_hat_temp

      update_start_time = time.time()
      for x in self.B_:
        self.theta_[x] = (1-c_hat_k_after)*theta_prime_tilde[x]+c_hat_k_after*theta_prime_hat[x]
      self.theta_[phi] += 4*(self.etahat_[phi]-eta_)
      self.compute_theta_perp()
      self.compute_P()
      kl=self.compute_KL()
      calc_KL_time +=  time.time() - update_start_time 
      print(iter_ ,":", "KL divergence: ", f'{kl:.16f}' ," time : %4.2f"% (time.time()-calc_KL_time-start),flush=True)



  def efficient_full_exact_ACDM(self, X, n_iter):
    """
    The implementation of efficient stable ACDM.
    Parameters 
    -----
    X : array-like, shape (n_samples,)
            Training vectors, where n_samples is the number of samples and 
            each element is a tuple of indices of 1s. 
    n_iter
    step 
    """
    calc_KL_time = 0
    e = np.e 
    B_size = len(self.B_)
    Z = len(self.S_)
    total_step = n_iter*B_size
    # Initialization of Theta
    theta_prime_tilde = dict()
    theta_prime_hat = dict()
    for phi in self.B_:
      theta_prime_tilde[phi] = 0.0
      theta_prime_hat[phi] = 0.0

    # Initialization of U
    U_hat = dict() 
    U_prime_hat = dict() 
    U_prime_tilde = dict()
    for x in self.S_:
      U_hat[x] = 0
      U_prime_hat[x] = 0.0
      U_prime_tilde[x] = 0.0
    # Initialization of Coefficients
    alpha = [0] * (total_step+1)
    gamma = [0] * (total_step+2)
    c_hat = [0] * (total_step+2)
    gamma[-1] =1/(2.0) 
    c_hat[-1] = 3.0 + 2*math.sqrt(2.000)    
    for k in range(total_step+1):
      gamma_k_before = gamma[k-1]
      gamma_k = math.sqrt(1.0/(4.0)+gamma_k_before**2)+1.0/(2.0)
      gamma[k] = gamma_k
      alpha[k] =1/(gamma_k)
      c_hat[k] =  (1-alpha[k]) * c_hat[k-1]
    # Caching 
    e_U_hat_dict = dict() 
    for x in self.S_:
      e_U_hat_dict[x] = 1.0
    # Beginning of Iteration
    start = time.time()
    for iter_ in range(n_iter):
      for index in range(B_size):
        k = iter_*B_size + index
        phi = random.choice(self.B_)
        if self.etahat_[phi] >= 1.0 or self.etahat_[phi]== 0.0:
          continue
        # Caching
        c_hat_k = c_hat[k]
        c_hat_k_after = c_hat[k+1]
        alpha_k_after = alpha[k+1]
        gamma_k = gamma[k]

        # calculate G, Z
        G = 0 
        for x in self.Ssub_[phi]:
          G += e_U_hat_dict[x]
        eta_ = G/Z
        # update tilde_theta_prime, hat_theta_prime
        nabla = 4.0*(self.etahat_[phi] - eta_)
        nabla_exact = np.log((1.0 -eta_)* self.etahat_[phi]/(eta_*(1-self.etahat_[phi])))

        delta_tilde = nabla_exact/len(X)
        delta_hat = nabla_exact/len(X)
        step1 = gamma_k/B_size * delta_tilde
        step2 = 1.0/c_hat_k_after*((alpha_k_after-(1-c_hat_k_after))*gamma_k/B_size*delta_tilde+(1-alpha_k_after)*delta_hat)


        theta_prime_tilde[phi] += step1
        theta_prime_hat[phi] += step2 
        for x in self.Ssub_[phi]:
          U_prime_tilde[x] += step1
          U_prime_hat[x] += step2
          U_hat_temp = (1-c_hat_k_after)*U_prime_tilde[x]+c_hat_k_after*U_prime_hat[x]
          e_U_hat_temp = e ** U_hat_temp
          Z += e_U_hat_temp - e_U_hat_dict[x]
          e_U_hat_dict[x] = e_U_hat_temp

      update_start_time = time.time()
      for x in self.B_:
        self.theta_[x] = (1-c_hat_k_after)*theta_prime_tilde[x]+c_hat_k_after*theta_prime_hat[x]
      self.theta_[phi] += 4*(self.etahat_[phi]-eta_)
      self.compute_theta_perp()
      self.compute_P()
      kl=self.compute_KL()
      calc_KL_time +=  time.time() - update_start_time 
      print(iter_ ,":", "KL divergence: ", f'{kl:.16f}' ," time : %4.2f"% (time.time()-calc_KL_time-start),flush=True)


  def check_two_dimesion_converge(self, X, n_iter):
    # Different from full_exact_acdm
    num_samplings = 2
    #------------------
    e = np.e 
    B_size = len(self.B_)
    Z = len(self.S_)
    total_step = n_iter*B_size
    # Initialization of Theta
    theta_prime_tilde = dict()
    theta_prime_hat = dict()
    for phi in self.B_:
      theta_prime_tilde[phi] = 0.0
      theta_prime_hat[phi] = 0.0

    # Initialization of U
    U_hat = dict() 
    U_prime_hat = dict() 
    U_prime_tilde = dict()
    for x in self.S_:
      U_hat[x] = 0
      U_prime_hat[x] = 0.0
      U_prime_tilde[x] = 0.0
    # Initialization of Coefficients
    alpha = [0] * (total_step+1)
    gamma = [0] * (total_step+2)
    c_hat = [0] * (total_step+2)
    gamma[-1] =1/(2.0) 
    c_hat[-1] = 3.0 + 2*math.sqrt(2.000)    
    for k in range(total_step+1):
      gamma_k_before = gamma[k-1]
      gamma_k = math.sqrt(1.0/(4.0)+gamma_k_before**2)+1.0/(2.0)
      gamma[k] = gamma_k
      alpha[k] =1/(gamma_k)
      c_hat[k] =  (1-alpha[k]) * c_hat[k-1]
    # Caching 
    e_U_hat_dict = dict() 
    for x in self.S_:
      e_U_hat_dict[x] = 1.0
    # Beginning of Iteration
    """
    ここが通常のものと違う。
    """
    import random
    parameters_to_optimize = random.sample(self.B_,num_samplings)
    self.parameters = parameters_to_optimize
    for iter_ in range(n_iter):
      for index in range(B_size):
        k = iter_*B_size + index
        """
        ここも違う
        """
        phi = parameters_to_optimize[k%2]
        if self.etahat_[phi] >= 1.0 or self.etahat_[phi]== 0.0:
          continue
        # Caching
        c_hat_k = c_hat[k]
        c_hat_k_after = c_hat[k+1]
        alpha_k_after = alpha[k+1]
        gamma_k = gamma[k]

        # calculate G, Z
        G = 0 
        for x in self.Ssub_[phi]:
          G += e_U_hat_dict[x]
        eta_ = G/Z
        # update tilde_theta_prime, hat_theta_prime
        nabla = 4.0*(self.etahat_[phi] - eta_)
        nabla_exact = np.log((1.0 -eta_)* self.etahat_[phi]/(eta_*(1-self.etahat_[phi])))

        delta_tilde = nabla_exact/len(X)
        delta_hat = nabla_exact/len(X)
        step1 = gamma_k/B_size * delta_tilde
        step2 = 1.0/c_hat_k_after*((alpha_k_after-(1-c_hat_k_after))*gamma_k/B_size*delta_tilde+(1-alpha_k_after)*delta_hat)

        theta_prime_tilde[phi] += step1
        theta_prime_hat[phi] += step2 
        for x in self.Ssub_[phi]:
          U_prime_tilde[x] += step1
          U_prime_hat[x] += step2
          U_hat_temp = (1-c_hat_k_after)*U_prime_tilde[x]+c_hat_k_after*U_prime_hat[x]
          e_U_hat_temp = e ** U_hat_temp
          Z += e_U_hat_temp - e_U_hat_dict[x]
          e_U_hat_dict[x] = e_U_hat_temp
      for x in self.B_:
        self.theta_[x] = (1-c_hat_k_after)*theta_prime_tilde[x]+c_hat_k_after*theta_prime_hat[x]
      self.theta_[phi] += 4*(self.etahat_[phi]-eta_)
    
      param_dict = dict()
      for param in parameters_to_optimize:
            param_dict[param] = self.theta_[param]
      self.param_value_list.append(param_dict)
        
      self.compute_theta_perp()
      self.compute_P()
      kl=self.compute_KL()

  def check_two_dimesion_converge_points(self, X, n_iter):
    # Different from full_exact_acdm
    num_samplings = 2
    #------------------
    e = np.e 
    B_size = len(self.B_)
    Z = len(self.S_)
    total_step = n_iter*B_size
    # Initialization of Theta
    theta_prime_tilde = dict()
    theta_prime_hat = dict()
    for phi in self.B_:
      theta_prime_tilde[phi] = 0.0
      theta_prime_hat[phi] = 0.0

    # Initialization of U
    U_hat = dict() 
    U_prime_hat = dict() 
    U_prime_tilde = dict()
    for x in self.S_:
      U_hat[x] = 0
      U_prime_hat[x] = 0.0
      U_prime_tilde[x] = 0.0
    # Initialization of Coefficients
    alpha = [0] * (total_step+1)
    gamma = [0] * (total_step+2)
    c_hat = [0] * (total_step+2)
    gamma[-1] =1/(2.0) 
    c_hat[-1] = 3.0 + 2*math.sqrt(2.000)    
    for k in range(total_step+1):
      gamma_k_before = gamma[k-1]
      gamma_k = math.sqrt(1.0/(4.0)+gamma_k_before**2)+1.0/(2.0)
      gamma[k] = gamma_k
      alpha[k] =1/(gamma_k)
      c_hat[k] =  (1-alpha[k]) * c_hat[k-1]
    # Caching 
    e_U_hat_dict = dict() 
    for x in self.S_:
      e_U_hat_dict[x] = 1.0
    # Beginning of Iteration
    """
    ここが通常のものと違う。
    """
    import random
    parameters_to_optimize = random.sample(self.B_,num_samplings)
    self.params
    for iter_ in range(n_iter):
      for index in range(B_size):
        k = iter_*B_size + index
        """
        ここも違う
        """
        phi = parameters_to_optimize[k%2]
        if self.etahat_[phi] >= 1.0 or self.etahat_[phi]== 0.0:
          continue
        # Caching
        c_hat_k = c_hat[k]
        c_hat_k_after = c_hat[k+1]
        alpha_k_after = alpha[k+1]
        gamma_k = gamma[k]

        # calculate G, Z
        G = 0 
        for x in self.Ssub_[phi]:
          G += e_U_hat_dict[x]
        eta_ = G/Z
        # update tilde_theta_prime, hat_theta_prime
        nabla = 4.0*(self.etahat_[phi] - eta_)
        nabla_exact = np.log((1.0 -eta_)* self.etahat_[phi]/(eta_*(1-self.etahat_[phi])))

        delta_tilde = nabla_exact/len(X)
        delta_hat = nabla_exact/len(X)
        step1 = gamma_k/B_size * delta_tilde
        step2 = 1.0/c_hat_k_after*((alpha_k_after-(1-c_hat_k_after))*gamma_k/B_size*delta_tilde+(1-alpha_k_after)*delta_hat)

        theta_prime_tilde[phi] += step1
        theta_prime_hat[phi] += step2 
        for x in self.Ssub_[phi]:
          U_prime_tilde[x] += step1
          U_prime_hat[x] += step2
          U_hat_temp = (1-c_hat_k_after)*U_prime_tilde[x]+c_hat_k_after*U_prime_hat[x]
          e_U_hat_temp = e ** U_hat_temp
          Z += e_U_hat_temp - e_U_hat_dict[x]
          e_U_hat_dict[x] = e_U_hat_temp
      for x in self.B_:
        self.theta_[x] = (1-c_hat_k_after)*theta_prime_tilde[x]+c_hat_k_after*theta_prime_hat[x]
      self.theta_[phi] += 4*(self.etahat_[phi]-eta_)
      self.compute_theta_perp()
      self.compute_P()
      kl=self.compute_KL()
      parameters_strings = [self.theta_[p]  for p in parameters_to_optimize]
      print(iter_ ," " ,parameters_strings," ",f'{kl:.16f}',flush=True)