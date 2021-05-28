from lib import *
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import rc  
import matplotlib.transforms as mtrans

def filename(targets, mw, k=10, run=0):
    return "accuracies_"+targets+"_mw_"+str(mw)+"k_"+str(k)+"_run_"+str(run)+".pickle"

def main():

    # run parameters 
    runs = 10
    ks = [0,2,4,6,8,10]
    mws = [0.01*i for i in range(21)]
    targets = "hot" 
    # targets = "cold"

    # # # # #
    heatmap = np.zeros((runs, len(mws), len(ks)))
    
    for run in range(runs):
        kidx, mwidx = 0, 0
        for k in ks:
            mwidx = 0
            for mw in mws:
                accuracies = pickle.load(open(filename(targets, mw, k), "rb"))
                heatmap[run, mwidx, kidx] = np.amax(accuracies)
                mwidx += 1
            kidx += 1

    # mean heatmap
    heatmap_2d = np.mean(heatmap, axis = 0) # avg over runs to give 2-array

    # # stdev heatmap
    # heatmap_2d = np.std(heatmap, axis = 0)/np.mean(heatmap, axis = 0)
    

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

    ax = sns.heatmap(heatmap_2d, cmap = "mako")
    ax.invert_yaxis()

    # # # # #
    plt.ylabel(lm("\mu_{W^{1}}"),fontsize=label_size)
    plt.tick_params(labelsize=tick_size)


    # the following are cosmetic changes for the plots 

    # ks_labels = [lm(str(ks[i])) for i in range(len(ks))]
    # plt.xticks([0,1,2,3,4,5], ks_labels)

    # labels = ["" for _ in range(len(mws))]
    # for i in range(3):
    #     labels[10*i] = lm(str(mws[10*i]))
    
    # print(labels)
    # plt.yticks([i for i in range(len(mws))], labels)

    # trans = mtrans.Affine2D().translate(0, 10)
    # for t in ax.get_yticklabels():
    #     t.set_transform(t.get_transform()+trans)

    # trans2 = mtrans.Affine2D().translate(25, 0)
    # for t in ax.get_xticklabels():
    #     t.set_transform(t.get_transform()+trans2)
    # plt.xlabel(lm("k"),fontsize=label_size)

    ax.set_title(lt("Accuracy vs. Symmetry Breaking"), y=1.1, pad=20, fontsize=title_size)

    plt.tight_layout()
    import datetime
    plt.savefig("heatmap_symmetry_breaking_parameters.pdf",bbox_inches='tight')
    plt.figure()
    # plt.show()



if __name__ == '__main__':
    main()