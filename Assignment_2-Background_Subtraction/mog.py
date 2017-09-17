import numpy as np
class MOG(object):
  def __init__(self,frame, K=3,T=0.3,lr=0.25):
    self.rows = frame.shape[0]
    self.cols = frame.shape[1]
    self.N = frame.shape[2]
    self.K = K
    self.T = T
    self.mean = np.zeros((self.rows,self.cols,K,self.N),dtype=np.float32)
    self.std_dev = np.ones((self.rows,self.cols,K),dtype=np.float32)
    self.weight = np.ones((self.rows,self.cols,K),dtype=np.float32)/K
    self.dist_mult = 2.5
    self.lr = lr
    self.HIGH_VAR=10
    self.denom_const = np.power(2*np.pi,(self.N)/2)

  def apply(self,frame):
    ## calculating mahalanaobis distance for each pixel
    sort_weights = self.weight/self.std_dev
    index_sort_weights = np.argsort(-sort_weights,axis=2)

    sqdist_frames = (np.sum((np.expand_dims(frame,axis=2)-self.mean)**2,axis=3))/self.std_dev
    prob_frames = np.exp((-0.5)*sqdist_frames)/(self.denom_const*(np.sqrt(self.N)*self.std_dev))  
    dist_frames = np.sqrt(sqdist_frames)
    admissible_frames = dist_frames < self.dist_mult*self.std_dev
    
    lin_indx = np.reshape(index_sort_weights,[-1])
    row_indx = np.repeat(np.arange(self.rows),self.cols*self.K)
    col_indx = np.tile(np.repeat(np.arange(self.cols),self.K),self.rows)
    sorted_wgts = np.reshape(self.weight[row_indx,col_indx,lin_indx],(self.rows,self.cols,-1))
    background_dist = np.cumsum(sorted_wgts,axis=2)<self.T
    is_background = np.any(np.logical_and(background_dist,admissible_frames),axis=2)
    result = (np.logical_not(is_background)*255).astype(np.uint8)
    
    modify_mask = np.any(admissible_frames,axis=2)
    matched_gaussian = np.choose(np.argmax(admissible_frames,axis=2),index_sort_weights.transpose(2,0,1))
    
    self.weight -= (np.expand_dims(modify_mask,axis=2)*self.weight)*(self.lr)
    row_indx = np.repeat(np.arange(self.rows),self.cols)
    col_indx = np.tile((np.arange(self.cols)),self.rows)
    lin_indx = np.reshape(matched_gaussian,[-1])
    self.weight[row_indx,col_indx,lin_indx] = np.reshape(modify_mask,[-1])*self.lr
    
    rho = np.reshape(modify_mask,[-1])*self.lr*(prob_frames[row_indx,col_indx,lin_indx] )
    self.mean[row_indx,col_indx,lin_indx,:] -= np.expand_dims(rho,axis=2)*self.mean[row_indx,col_indx,lin_indx,:]
    self.mean[row_indx,col_indx,lin_indx,:] += np.expand_dims(rho,axis=2)*frame[row_indx,col_indx,:]
    
    prev_var = (self.std_dev[row_indx,col_indx,lin_indx]*np.reshape(modify_mask,[-1]))**2
    add_term = rho*np.sum((frame[row_indx,col_indx,:]-self.mean[row_indx,col_indx,lin_indx,:])**2,axis=1)
    new_var = (1-rho)*prev_var + add_term
    self.std_dev[row_indx,col_indx,lin_indx] = np.sqrt(new_var)
    
    
    ## Adding new gaussian
    last_idx = index_sort_weights[:,:,-1]
    lin_indx = np.reshape(last_idx,[-1])
    updation_mask = np.logical_not(modify_mask)
    new_mean_values = (self.mean - (np.expand_dims(frame,axis=2)))[row_indx,col_indx,lin_indx,:]
    lin_mask = np.reshape(updation_mask,[-1])
    self.mean[row_indx,col_indx,lin_indx,:] -= np.expand_dims(lin_mask,axis=3)*new_mean_values
    self.std_dev[row_indx,col_indx,lin_indx] += lin_mask*self.HIGH_VAR ## CHECK THIS
#     result = np.zeros(frame.shape[:2],dtype=np.uint8)
#     for i in range(frame.shape[0]):
#       for j in range(frame.shape[1]):
#         indices = index_sort_weights[i][j]
#         values = self.weight[indices,i,j]
#         background_dist = np.cumsum(values)<self.T
#         admissible = admissible_frames[i][j]
#         is_background = np.logical_and(background_dist,admissible)
#         idx = np.argmax(is_background)
#         if (is_background[idx]):
#           result[i][j] = 255

#         if (not (np.any(admissible))):
#           #ADD NEW GAUSSIAN
#           idx = indices[-1]
#           self.mean[idx][i][j] = frame[i][j]
#           self.std_dev[idx][i][j] = self.HIGH_VAR
#         else:
#           rho = self.lr*(prob_frames[idx][i][j])

#           self.mean[idx][i][j] *= (1-rho)
#           self.mean[idx][i][j] += frame[i][j]*rho
#           variance = (1-rho)*(self.std_dev[idx][i][j]**2) + rho*(np.sum((frame[i][j]-self.mean[idx][i][j])**2))
#           self.std_dev[idx][i][j] = np.sqrt(variance)
    return result

#class MOG(object):
#  def __init__(self,frame, K=5,T=0.25,lr=0.2):
#    self.rows = frame.shape[0]
#    self.cols = frame.shape[1]
#    self.N = frame.shape[2]
#    self.K = K
#    self.T = T
#    self.mean = [np.zeros(frame.shape,dtype=np.float32) for _ in K]
#    self.std_dev = [np.ones(frame.shape[:2],dtype=np.float32) for _ in K]
#    self.weight = np.array([np.ones(frame.shape[:2],dtype=np.float32)/K for _ in K])
#    self.dist_mult = 2.5
#    self.lr = lr
#    self.HIGH_VAR=10
#    self.denom_const = np.power(2*np.pi,(self.N)/2)
#
#  def apply(self,frame):
#    ## calculating mahalanaobis distance for each pixel
#    sort_weights = np.stack([self.weight[i]/self.std_dev[i] for i in range(self.K)],axis=2)
#    index_sort_weights = np.argsort(-sort_weights,axis=2)
#
#    sqdist_frames=[np.sum((frame-self.mean[i])**2,axis=2)/self.std_dev[i] for i in range(self.K)]
#    prob_frames=[np.exp((-0.5)*sqdist_frames[i])/(self.denom_const*(np.sqrt(self.N)*self.std_dev[i])) for i in range(self.K)]
#    dist_frames = np.sqrt(sqdist_frames)
#    admissible_frames = np.stack([dist_frames < self.dist_mult*std_dev[i] for i in range(self.K)])
#
#    result = np.zeros(frame.shape[:2],dtype=np.uint8)
#    for i in range(frame.shape[0]):
#      for j in range(frame.shape[1]):
#        k = 0
#        cum = 0
#        indices = index_sort[i][j]
#        values = self.weight[indices,i,j]
#        background_dist = np.cumsum(values)<self.T
#        admissible = admissible_frames[i][j]
#        is_background = np.logical_and(background_dist,admissible)
#        idx = np.argmax(is_background)
#        if (is_background[idx]):
#          result[i][j] = 255
#
#        if (not (np.all(admissible))):
#          #ADD NEW GAUSSIAN
#          idx = indices[-1]
#          self.mean[idx][i][j] = frame[i][j]
#          self.std_dev[idx][i][j] = self.HIGH_VAR
#        else:
#          idx = np.argmax(admissible)
#          self.weight[:,i,j] *= (1-self.lr)
#          self.weight[idx,i,j] += (self.lr)
#          rho = self.lr*(prob_frames[idx][i][j])
#
#          self.mean[i][j] *= (1-rho)
#          self.mean[i][j] += frame[i][j]*rho
#          variance = (1-rho)*(self.std_dev[idx][i][j]**2) + rho*(np.sum((pixel-self.mean[idx][i][j])**2))
#          self.std_dev[idx][i][j] = np.sqrt(variance)
#    return result
#
#  def process_pixel(self,row,col,pixel):
#    #sort in decreasing order of weight/std_dev
#    sorted(self.params[row][col],key = lambda k: (-1*(k['weight']/k['std_dev'])))
#    params = self.params[row][col]
#
#    last_bg_idx = 0
#    cum_weight = 0
#    for param in params:
#      cum_weight += param["weight"]
#      if(cum_weight > self.T):
#        break
#      last_bg_idx += 1
#    
#    # Estimation Stage
#    sq_dist = np.array([self.mahalanaobis_dist_sq(pixel,param["mean"],param["std_dev"]) for param in self.params[row][col]])
#    prob = [self.prob_pixel(sq_dist[i],params[i]["std_dev"]) for i in range(len(params))]
#
#    match_idx = len(params)
#    for i in range(len(params)):
#      if(np.sqrt(sq_dist[i]) < self.dist_mult*params[i]["std_dev"]):
#        match_idx = i
#        break
#
#    #Updation Stage
#    if match_idx==len(params):
#      min_idx = np.argmin(prob)
#      params[min_idx]["mean"] = pixel
#      params[min_idx]["std_dev"] = self.HIGH_VAR
#    else:
#      for (idx,param) in enumerate(params):
#        if idx==match_idx:
#          param["weight"] = (1-self.lr)*param["weight"] + self.lr
#          rho = self.lr * (prob[match_idx])
#          param["mean"] = (1-rho)*param["mean"] + rho*pixel
#          variance = (1-rho)*(param["std_dev"]**2) + rho*(np.sum((pixel-param["mean"])**2))
#          param["std_dev"] = np.sqrt(variance)
#        else:
#          param["weight"] = (1-self.lr)*param["weight"]
#
#    return (match_idx <= last_bg_idx)
#
#  def apply(self,frame):
#    result = np.zeros(frame.shape[:2],dtype=np.uint8)
#    N = frame.shape[2]
#    for row in range(frame.shape[0]):
#      for col in range(frame.shape[1]):
#        result[row][col] = (0 if self.process_pixel(row,col,frame[row][col]) else 255)
#    return result
