from lib import *
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=10.**-3, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--mw', type=float, default=0.0, metavar='M',
                        help='Mean of the final weights')
    parser.add_argument('--run', type=int, default=0, metavar='R',
                        help='for parellization')
    parser.add_argument('--targets', type=str, default="hot")
    parser.add_argument('--k', type=int, default=10, metavar='K')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


    class ReLUNet(nn.Module):
        def __init__(self, args, width=50, bias=True):
            super(ReLUNet, self).__init__()
            self.args = args
            
            self.input = nn.Linear(28*28,width,bias=bias)
            torch.nn.init.normal_(self.input.weight,mean=0.0,std=1.0/math.sqrt(self.input.in_features))
            
            self.output = nn.Linear(width,10,bias=bias)
            torch.nn.init.normal_(self.output.weight,mean=0,std=1.0/math.sqrt(self.output.in_features))
            torch.nn.init.normal_(self.output.weight[0:args.k],mean=args.mw,std=1.0/math.sqrt(self.output.in_features))

        def forward(self,x):
            x = x.view(x.shape[0],x.shape[1],x.shape[2]*x.shape[3])
            x = self.input(x) # linear
            x = F.relu(x) # relu
            x = self.output(x) # linear
            x = x.view(x.shape[0],x.shape[2])
            return x


    def train(args, model, device, train_loader, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            
            ones = torch.sparse.torch.eye(10).to(device)  
            target = ones.index_select(0, target) # one-hot
            
            if args.targets == "cold":
                target = torch.ones((10,)) - target # one-cold

            loss = F.mse_loss(output, target)
            
            loss.backward()
            optimizer.step()
            if batch_idx % 100*args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))


    def test(model, device, test_loader, epoch, accuracies, args, quiet=False, vals=True):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                target_int = target # easier to count correct predictions 

                ones = torch.sparse.torch.eye(10).to(device)  
                target = ones.index_select(0, target) # one-hot

                if args.targets == "cold":
                    target = torch.ones((10,)) - target # one-cold
                    pred = output.argmin(dim=1, keepdim=True) # one-cold wants the lowest value prediction
            
                test_loss += F.mse_loss(output, target, reduction='sum').item()
                
                if args.targets == "hot":
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target_int.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        
        if quiet == False:
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))
        if vals == True:
            acc = 100. * correct / len(test_loader.dataset)
            accuracies[epoch-1] = acc
            print("accuracy for epoch ", epoch-1)
            return acc


    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../data', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)



    def train_loop(model, accuracies):
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            test(model, device, test_loader, epoch, accuracies, args)
            scheduler.step()


    accuracies = np.zeros((args.epochs))

    print("Run # ", args.run)
    print("one-", args.targets)

    model = ReLUNet(args, width=50, bias=False).to(device)
    train_loop(model, accuracies)

    targets = "hot"
    if args.targets == "cold":
        targets = "cold"

    filename = "accuracies_"+targets+"_mw_"+str(args.mw)+"k_"+str(args.k)+"_run_"+str(args.run)+".pickle"
    pickle.dump(accuracies, open(filename, "wb"))
    print("Saved accuracy list as ", filename)

if __name__ == "__main__":
    main()