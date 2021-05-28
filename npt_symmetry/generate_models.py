import sys
sys.path.append("./")
sys.path.append("..")
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
    runs = 1

    if args.d_in == 1:
        if args.activation == "GaussNet":
            xs = 0.01*torch.tensor([[-1],[-0.6],[-0.2],[0.2],[0.6], [1.0]])
            xset = "xset2"

    args.n_inputs = len(xs)

    for run in range(runs):
        print("Generating networks for "+args.activation+" at width "+str(args.width), " - run ", run+1, " of ", runs)
        fss = create_networks(xs, args)
        pickle.dump(fss, open("run"+str(run)+"_dout="+str(args.d_out)+"_"+args.activation+"_1e"+str(int(np.log10(args.n_models)))+"models_"+str(args.width)+"width_"+xset+".pickle",'wb'))

if __name__ == "__main__":
    main()