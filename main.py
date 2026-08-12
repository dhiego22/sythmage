import argparse
from train import train
from inference import reconstruct

def main()-> None:
    parser = argparse.ArgumentParser(prog='sythmage', description='Synthetic 3D medical images translator')
    sub = parser.add_subparsers(dest='cmd', required= True)

    parser_train = sub.add_parser('train', help='Train model')
    parser_train.add_argument('--model', default = 'gan3D')
    parser_train.add_argument('--dir-base', required=True)
    parser_train.add_argument('--dir-target', required=True)
    parser_train.add_argument('--epochs', type=int, default=10)
    parser_train.add_argument('--batch-size', type=int, default=1)
    parser_train.add_argument('--lr', type=float, default=2e-4)
    parser_train.add_argument('--lambda-r1', type=float,default=10)
    parser_train.add_argument('--patch-size', type=int, nargs=3, default=[64,64,64])
    parser_train.add_argument('--patches-per-volume', type=int, default=16)
    parser_train.add_argument('--metrics-dir', default='metrics')
    parser_train.add_argument('--checkpoints-dir', default='checkpoints')
    parser_train.add_argument('--samples-dir', default='samples')
    

    p_infer = sub.add_parser('reconstruct', help='Synthesize files and compute metrics')
    p_infer.add_argument('--checkpoint', required=True)
    p_infer.add_argument('--in-dir', required=True)
    p_infer.add_argument('--out-dir', required=True)
    p_infer.add_argument('--patch-size', type=int, nargs=3, default=[64,64,64])
    p_infer.add_argument('--overlap', type=int, nargs=3, default=[32,32,32])
        
    
    args = parser.parse_args()
    if args.cmd == 'train':
        train(
            dir_base = args.dir_base,
            dir_target = args.dir_target,
            epochs = args.epochs,
            batch_size = args.batch_size,
            lr = args.lr,
            lambda_r1 = args.lambda_r1,
            patch_size = (64,64,64),
            patches_per_volume = 32,
            save_dir = args.checkpoints_dir,
            
            
        )
    elif args.cmd == 'reconstruct':
        reconstruct(
            checkpoint_path=args.checkpoint,
            in_dir=args.in_dir,
            out_dir=args.out_dir,
            patch_size=tuple(args.patch_size),
            overlap=tuple(args.overlap),
            
        )
    else:
        print('ERROR')   
    
if __name__ == '__main__':
    main()
