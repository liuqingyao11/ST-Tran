import argparse
def getparse():
    parse = argparse.ArgumentParser()
    parse.add_argument('-height', type=int, default=100)
    parse.add_argument('-width', type=int, default=100)
    parse.add_argument('-traffic', type=str, default='call')
    #model
    parse.add_argument('-close_size', type=int, default=3)#
    parse.add_argument('-period_size', type=int, default=3)#
    parse.add_argument('-trend_size', type=int, default=0)#
    parse.add_argument('-test_size', type=int, default=24*7)
    parse.add_argument('-p_model_d',type=int,default=128)#
    parse.add_argument('-t_model_d',type=int,default=64)#
    parse.add_argument('-c_model_d',type=int,default=256)
    parse.add_argument('-s_model_d',type=int,default=64)
    parse.add_argument('-model_N',type=int,default=6)
    parse.add_argument('-k',type=int,default=20)
    parse.add_argument('-spatial',type=str,choices=['gcn','transformer'],help='choose the spatial model type',default='transformer')
    parse.add_argument('-mode',type=str,default='corr',choices=['cos','corr'],help='choose the way to get adj metrix') 
    parse.add_argument('-c',action='store_true')
    parse.add_argument('-s',action='store_true')
    parse.add_argument('-FS',action='store_true')
    parse.add_argument('-nb_flow', type=int, default=2)
    parse.add_argument('-flow',type=int,choices=[0,1],default=0,help='in--0,out--1')
    parse.add_argument('-c_t',type=str,default='p',choices=['t','p','tp','c','r'])
    parse.add_argument('-s_t',type=str,default='c',choices=['t','p','tp','c','r'])
    #training
    parse.add_argument('-train', dest='train', action='store_true')
    parse.add_argument('-no-train', dest='train', action='store_false')
    parse.set_defaults(train=True)
    parse.add_argument('-rows', nargs='+', type=int, default=[40, 60])
    parse.add_argument('-cols', nargs='+', type=int, default=[40, 60])
    parse.add_argument('-loss', type=str, default='l2', help='l1 | l2')
    parse.add_argument('-lr', type=float)
    parse.add_argument('-batch_size', type=int, default=64, help='batch size')
    parse.add_argument('-se',type=int)
    parse.add_argument('-epoch_size', type=int, default=500, help='epochs')
    parse.add_argument('-test_row', type=int, default=51, help='test row')
    parse.add_argument('-test_col', type=int, default=60, help='test col')
    parse.add_argument('-g',type=str,default=None)
    parse.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parse.add_argument('-w',type=str)
    parse.add_argument('-save_dir', type=str, default='results')
    parse.add_argument('-best_valid_loss',type=float,default=1)
    parse.add_argument('-lr-scheduler', type=str, default='poly',choices=['poly', 'step', 'cos'],
                            help='lr scheduler mode: (default: poly)')

    parse.add_argument('-warmup',type=int,default=100)
    parse.add_argument('-test_batch_size',type=int,default=1)

    return parse.parse_args()