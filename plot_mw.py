from lib import *
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import rc  

def filename(targets, mw, k=10, run=0):
    return "accuracies_"+targets+"_mw_"+str(mw)+"k_"+str(k)+"_run_"+str(run)+".pickle"


def main():

    # run paramters:
    runs = 20
    mws = [0.01*i for i in range(21)]

    targets = "hot" 
    # targets = "cold"
    # # # # #

    acc_list, mw_vals = [], []

    for run in range(runs): # seaborn automatically averages over runs when plotting
        for mw in mws:
            accuracies = pickle.load(open(filename(targets, mw, 10, run), "rb")) # k = 10 is when all means are mw
            acc_list.append(np.amax(accuracies))
            mw_vals.append(mw)

    df = pd.DataFrame({"mw_vals": mw_vals, "accuracies": acc_list})


    sns.set_palette("husl", 8)
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    rc('text', usetex=True)
    fsize = 24
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    plt.rc('font', size=fsize)  # controls default text sizes
    plt.rc('axes', titlesize=20)  # fontsize of the axes title
    plt.rc('axes', labelsize=20)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=18)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=18)  # fontsize of the tick labels
    plt.rc('legend', fontsize=20)  # legend fontsize
    plt.rc('figure', titlesize=fsize)  # fontsize of the figure title

    sns.set_style("ticks", {"xtick.major.size":18,
    "ytick.major.size":18})
    def lt(s):
        return (r'$\mathrm{' + s + r'}$').replace(" ", "\,\,")

    def lm(s):
        return r'$' + s + r'$'
    title_size, tick_size = fsize, fsize
    label_size = fsize
    sns.set_style(style="darkgrid")

    sns.lineplot(data=df, x="mw_vals", y="accuracies")

    plt.title(lt("One\\text{-}cold Accuracy vs. Init. Mean"), y=1.1, fontsize=title_size)

    # # # # #
    plt.xlabel(lm("\mu_{W^{1}}"),fontsize=label_size)
    plt.tick_params(labelsize=tick_size)
    plt.ylabel(lt("Max Accuracy")+lm(" (\%)"),fontsize=label_size)
    plt.tight_layout()
    plt.savefig("onecold.pdf",bbox_inches='tight')
    plt.figure()
    # plt.show() 



if __name__ == '__main__':
    main()