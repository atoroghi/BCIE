from trainer import Trainer
from tester import Tester
from dataload import LoadDataset
from measure import Measure
from recommender import Recommender
from updater import Updater
from plotter import Plotter
import matplotlib.pyplot as plt
import torch, argparse, time, os
from torch.distributions.multivariate_normal import MultivariateNormal

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-test_name', default=None, type=str, help="folder for test results")
    parser.add_argument('-ne', default=100, type=int, help="number of epochs")
    parser.add_argument('-ni', default=0, type=float, help="noise intensity")
    parser.add_argument('-lr', default=1, type=float, help="learning rate")
    parser.add_argument('-reg_lambda', default=0.03, type=float, help="l2 regularization parameter")
    parser.add_argument('-dataset', default="ML_FB", type=str, help="wordnet dataset")
    parser.add_argument('-emb_dim', default=200, type=int, help="embedding dimension")
    parser.add_argument('-save_each', default=50, type=int, help="validate every k epochs")
    parser.add_argument('-max_iters_laplace', default=1000, type=int, help="Maximum number of iterations for Laplace Approximation")
    parser.add_argument('-alpha', default=0.01, type=float, help="Learning rate for Laplace Approximation")
    parser.add_argument('-etta', default=1.0, type=float, help="Learning rate for Laplace Approximation")
    
    parser.add_argument('-workers', default=8, type=int, help="Number of workers for dataloader")
    parser.add_argument('-batch_size', default=16384, type=int, help="batch size")
    parser.add_argument('-neg_ratio', default=10, type=int, help="number of negative examples per positive example")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    # ensure test_name exits, make test result folder
    if args.test_name == None:
        raise ValueError("enter a test name folder using -test_name")
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("results", args.test_name), exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # print important hyperparameters
    print("epochs: {}, batch size: {}, dataset: {}, device: {}". format(
          args.ne, args.batch_size, args.dataset, device
    ))

    print("loading data")
    dataset = LoadDataset('train', args)
    
    print("training")
    trainer = Trainer(dataset, args, device)
    trainer.train()

    print("~~~~ Select best epoch on validation set ~~~~")
    epochs2test = [str(int(args.save_each * (i + 1))) for i in range(args.ne // args.save_each)]
    dataset = Dataset(args.dataset, args.ni)
    
    best_mrr = -1.0
    best_epoch = "0"
    for epoch in epochs2test:
        start = time.time()
        print(epoch)
        model_path = "models/" + args.dataset + "/" + epoch + ".chkpnt"
        tester = Tester(dataset, model_path, "valid")
        mrr = tester.test()
        if mrr > best_mrr:
            best_mrr = mrr
            best_epoch = epoch
        print(time.time() - start)

    print("Best epoch: " + best_epoch)

    print("~~~~ Testing on the best epoch ~~~~")
    best_model_path = "models/" + args.dataset + "/" + best_epoch + ".chkpnt"
    tester = Tester(dataset, best_model_path, "test")
    tester.test()
