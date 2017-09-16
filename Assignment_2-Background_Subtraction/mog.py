import numpy as np

class MOG(object):
  def __init__(self,frame, K=5,T=0.25,lr=0.2):
    self.rows = frame.shape[0]
    self.cols = frame.shape[1]
    self.N = frame.shape[2]
    self.K = K
    self.T = T
    self.dist_mult = 2.5
    self.lr = lr
    self.params = [[[{"mean":np.zeros(frame.shape[2]),"std_dev": 10,"weight": (1.0/K)} for a in range(self.K)]  for b in range(self.cols)] for c in range(self.rows)]
    self.HIGH_VAR=10
    self.denom_const = np.power(2*np.pi,(self.N)/2)

  def mahalanaobis_dist_sq(self,pixel,mean,std_dev):
    arg = np.sum((pixel-mean)**2)/std_dev
    return arg
  def prob_pixel(self,sq_dist,std_dev):
    num = np.exp(-0.5*sq_dist)
    denom = self.denom_const*(np.sqrt(self.N)*std_dev)
    return num/denom

  def process_pixel(self,row,col,pixel):
    #sort in decreasing order of weight/std_dev
    sorted(self.params[row][col],key = lambda k: (-1*(k['weight']/k['std_dev'])))
    params = self.params[row][col]

    last_bg_idx = 0
    cum_weight = 0
    for param in params:
      cum_weight += param["weight"]
      if(cum_weight > self.T):
        break
      last_bg_idx += 1
    
    # Estimation Stage
    sq_dist = np.array([self.mahalanaobis_dist_sq(pixel,param["mean"],param["std_dev"]) for param in self.params[row][col]])
    prob = [self.prob_pixel(sq_dist[i],params[i]["std_dev"]) for i in range(len(params))]

    match_idx = len(params)
    for i in range(len(params)):
      if(np.sqrt(sq_dist[i]) < self.dist_mult*params[i]["std_dev"]):
        match_idx = i
        break

    #Updation Stage
    if match_idx==len(params):
      min_idx = np.argmin(prob)
      params[min_idx]["mean"] = pixel
      params[min_idx]["std_dev"] = self.HIGH_VAR
    else:
      for (idx,param) in enumerate(params):
        if idx==match_idx:
          param["weight"] = (1-self.lr)*param["weight"] + self.lr
          rho = self.lr * (prob[match_idx])
          param["mean"] = (1-rho)*param["mean"] + rho*pixel
          variance = (1-rho)*(param["std_dev"]**2) + rho*(np.sum((pixel-param["mean"])**2))
          param["std_dev"] = np.sqrt(variance)
        else:
          param["weight"] = (1-self.lr)*param["weight"]

    return (match_idx <= last_bg_idx)

  def apply(self,frame):
    result = np.zeros(frame.shape[:2],dtype=np.uint8)
    N = frame.shape[2]
    for row in range(frame.shape[0]):
      for col in range(frame.shape[1]):
        result[row][col] = (0 if self.process_pixel(row,col,frame[row][col]) else 255)
    return result
