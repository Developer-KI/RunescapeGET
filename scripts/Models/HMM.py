#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from hmmlearn.hmm import MultinomialHMM
#%%

def HMM(features,iter=100,window=100,diffpercent=0.1,n_components=3, selfselect:pd.array= None):

    n_features= features.shape[1]
    n_samples= features.shape[0]
    #startprob = np.array([.17,.66,.17]) #reasonable to keep up/down always less than sideways
    #transprob = np.array([[.17,.66,.17],[.17,.66,.17],[.17,.66,.17]])
    #emissionprob = np.array([[.17,.66,.17],[.17,.66,.17],[.17,.66,.17]])
    #HMMmodel = MultinomialHMM(n_components=n_components, startprob_prior=startprob, transmat_prior=transprob, n_iter=iter) #leave init_params empty to self-select probabilities
    #HMMmodel.emissionprob_ = np.array(emissionprob)

    if selfselect==None:
        HMMmodel = MultinomialHMM(n_components=n_components, n_iter=iter) #leave init_params empty to self-select probabilities
        HMMmodel.fit(features)
        hidden_states= HMMmodel.predict(features)

        log_likelihood = HMMmodel.score(features)
        # Estimate number of parameters
        num_parameters = n_components**2 +(n_components*n_features) + n_components
        # Compute AIC & BIC
        aic = 2 * num_parameters - 2 * log_likelihood
        bic = num_parameters * np.log(n_samples) - 2 * log_likelihood

        print(f"AIC: {aic}, BIC: {bic}")
        return [aic,bic], HMMmodel

#%%
