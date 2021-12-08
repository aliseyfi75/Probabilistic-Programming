import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plots(samples, n):

    plt.switch_backend('agg')
    
    num_samples = len(samples)

    # this is ugly. Fix it. 
    for n in range(num_samples):
        samples[n] = np.array(samples[n], dtype=int)

    variables = np.array(samples,dtype=object).T.tolist()

    for d in range(len(variables)):
        plt.figure(figsize=(5,4))        
        plt.xlabel("theta["+str(d)+']')

        # plt.title("Distributions obtained program " + str(i) + " with " + str(num_samples) + " particles")

        sns.histplot(variables[d],kde=False, bins=50, stat="density")

        figstr = "histograms/n_"+str(num_samples)+"_theta_"+str(d)
        plt.savefig(figstr)

    # print("\n")
    # plt.show()
    # plt.close('all')
