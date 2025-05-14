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


# scores=[]
# total_iter= 
# for i in range(1,total_iter):
#     # tempAIC=[]
#     # tempBIC=[]  
#     for j in range(2,6):
#         HMMmodel = MultinomialHMM(n_components=n_components, n_iter=i) #leave init_params empty to self-select probabilities
#         HMMmodel.fit(X_encoded)
#         hidden_states= HMMmodel.predict(X_encoded)
#         log_likelihood = HMMmodel.score(X_encoded)
#         # Estimate number of parameters
#         num_parameters = n_components**2 +(n_components*X_encoded.shape[1]) + n_components
#         # Compute AIC & BIC
#         aic = 2 * num_parameters - 2 * log_likelihood
#         bic = num_parameters * np.log(X.shape[0]) - 2 * log_likelihood
#         # tempAIC.append(aic)
#         # tempBIC.append(bic)
#         scores.append({"Hidden States": j, "Iterations": total_iter, "AIC": aic, "BIC": bic})

#     # avg_aic = np.mean(tempAIC)
#     # avg_bic = np.mean(tempBIC)  

#     # scores[f'Run {i}']= [aic,bic]

#     if i == total_iter // 4:
#         print("25% completed...")
#     elif i == total_iter // 2:
#         print("50% completed...")
#     elif i == (3 * total_iter) // 4:
#         print("75% completed...")
#     elif i == total_iter:
#         print("100% done!")

# score_final = pd.DataFrame(scores) 
# #%%

# plt.figure(figsize=(10, 5))
# plt.plot(range(1,total_iter), score_final['AIC'], marker="o", markersize='1', linestyle="-", label="AIC")
# plt.plot(range(1,total_iter),score_final['AIC'], marker="o", markersize='1', linestyle="-", label="BIC")

# plt.xlabel("Iteration Count")
# plt.ylabel("AIC and BIC")
# plt.legend()
# plt.xticks(rotation=45)
# plt.grid()

# plt.show()

