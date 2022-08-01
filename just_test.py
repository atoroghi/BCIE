import torch, argparse, time, os, sys, yaml
from dataload import DataLoader
from tester import test

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-test_name', default='q_test')
    #parser.add_argument('-load_epoch', default=0)
    return parser.parse_args() 

# eval
if __name__ == '__main__':
    print('fix implimentation')
    sys.exit()
    
    args = get_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # get args from file
    with open(os.path.join('results', args.test_name, 'info.yml'), 'r') as f:
        yml = yaml.safe_load(f)
        for key in yml.keys():
            setattr(args, key, yml[key])
    
    # load model
    path = os.path.join('results', args.test_name)
    load_path = os.path.join(path, 'models', 'best_model.pt')
    model = torch.load(load_path).to(device)

    # load
    dataloader = DataLoader(args)

    test(model, dataloader, 5, args, 'test')

