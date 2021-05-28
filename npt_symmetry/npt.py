import sys
sys.path.append("./")
sys.path.append("..")
import argparse
from lib import *


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--activation", type=str, default = "GaussNet")
    parser.add_argument('--exp', type=str, default = None)
    parser.add_argument("--width", type=int, default = 1000)
    parser.add_argument("--n-inputs", type = int, default = 6)
    parser.add_argument("--n-models", type = int, default = 10**3)
    parser.add_argument("--d-in", type = int, default = 1)
    parser.add_argument("--d-out", type = int, default = 5)
    parser.add_argument("--sb", type = float, default = 1.0)
    parser.add_argument("--sw", type = float, default = 1.0)
    parser.add_argument("--mb", type = float, default = 0.0)
    parser.add_argument("--mw", type = float, default = 0.0)
    parser.add_argument("--cuda", action = 'store_true', default = False)

    args = parser.parse_args()

    widths = [1000]
    
    runs = 1 # runs per width, usually set to 10 or 1


    if args.d_in == 1:
        if args.activation == "GaussNet":
            xs = 0.01*torch.tensor([[-1],[-0.6],[-0.2],[0.2],[0.6], [1.0]])
            xset = "xset2"

    args.n_inputs = len(xs)

    for n in [2, 4]:

        fss = {}    # dictionary for storing outputs after importing
                    # keys are widths

        for width in widths:
            print("Unpickling width "+str(width))
            args.width = width
            for run in range(runs):
                with open("run"+str(run)+"_dout="+str(args.d_out)+"_"+args.activation+"_1e"+str(int(np.log10(args.n_models)))+"models_"+str(args.width)+"width_"+xset+".pickle",'rb') as handle:
                    if run == 0:
                        fss[width] = pickle.load(handle)
                    else:
                        fss[width] = torch.cat((fss[width], pickle.load(handle)))
        

        print("Computing "+str(n)+"-pt function for activation "+args.activation)

        fss_chunk = {}
        k = 10
        chunk = len(fss[widths[0]])//k
        print("Models in each chunk: ", chunk)

        widths_list, n_diff_full, backgrounds, n_exp = [], [], [], [0. for _ in range(10)]
        for width in widths:
            for chunk_num in range(10):
                # this is a dictionary (with keys = widths) for a single chunk
                fss_chunk[width] = fss[width].narrow_copy(0,chunk_num*chunk,chunk)
                n_tensor = torch.mean(n_point(fss_chunk[width], n), dim=0)
                n_exp[chunk_num] = n_tensor.tolist()
            pickle.dump(np.nanmean(n_exp, axis = 0),open(str(n)+"ptexp_dout"+str(args.d_out)+str(args.width)+".pickle","wb") )
            pickle.dump(np.nanstd(n_exp, axis = 0),open(str(n)+"ptexp_dout"+str(args.d_out)+str(args.width)+"stdev.pickle","wb") )

if __name__ == "__main__":
    main()