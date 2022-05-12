from trainer import Trainer
from tester import Tester
from dataload import LoadDataset
from measure import Measure
from recommender import Recommender
from updater import Updater
import matplotlib.pyplot as plt
import torch, argparse, time, os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-test_name', default=None, type=str, help="folder for test results")
    parser.add_argument('-ne', default=100, type=int, help="number of epochs")
    parser.add_argument('-ni', default=0, type=float, help="noise intensity")
    parser.add_argument('-lr', default=1, type=float, help="learning rate")
    parser.add_argument('-reg_lambda', default=0.03, type=float, help="l2 regularization parameter")
    parser.add_argument('-dataset', default="ML_FB", type=str, help="wordnet dataset")
    parser.add_argument('-emb_dim', default=200, type=int, help="embedding dimension")
    parser.add_argument('-neg_ratio', default=10, type=int, help="number of negative examples per positive example")
    parser.add_argument('-batch_size', default=16384, type=int, help="batch size")
    parser.add_argument('-save_each', default=50, type=int, help="validate every k epochs")
    parser.add_argument('-max_iters_laplace', default=1000, type=int, help="Maximum number of iterations for Laplace Approximation")
    parser.add_argument('-alpha', default=0.01, type=float, help="Learning rate for Laplace Approximation")
    parser.add_argument('-etta', default=1.0, type=float, help="Learning rate for Laplace Approximation")
    parser.add_argument('-workers', default=8, type=int, help="threads for dataloader")
    parser.add_argument('-num_run', default=1, type=float, help="number of run for saving")
    parser.add_argument('-neg_power', default=0, type=float, help="power for neg sampling disribution")
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = get_args()

    # ensure test_name exits, make test result folder
    if args.test_name == None:
        raise ValueError('enter a test name folder using -test_name')
    os.makedirs('results', exist_ok=True)
    os.makedirs(os.path.join("results", args.test_name), exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # print important hyperparameters
    print('epochs: {}, batch size: {}, dataset: {}, device: {}'. format(
          args.ne, args.batch_size, args.dataset, device
    ))

    print('loading data')
    dataset = LoadDataset('train', args)

    print('training')
    trainer = Trainer(dataset, args, device)
    trainer.train()

    # TODO: training and testing should be called seperately
            # ie: train script, then test script (call bash both.sh if we want)

    # TODO: trainer and tester should be functions not classes
            # too many variable names for no reason
            # also tester could just be merged with recommender...
            # only purpose is getting the embedding list

    print('testing')
    # temp!!!! take away the -1 before push to github
    model_path = 'models/' + args.dataset + '/' + str(args.ne - 1) + '.chkpnt'
    dataset = LoadDataset('test', args)
    tester = Tester(model_path, 'valid', dataset, args)
    
    hone, hthree, hfive, hten, htwenty = tester.evaluate_precritiquing()
    print('pre-critiquing hit@1:')
    print(hone)
    print('pre-critiquing hit@3:')
    print(hthree)
    print('pre-critiquing hit@5:')
    print(hfive)
    print('pre-critiquing hit@10:')
    print(hten)
    print('pre-critiquing hit@20:')
    print(htwenty)