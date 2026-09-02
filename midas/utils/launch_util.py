from midas.utils.general_utils import AttrDict

def parse_training_args(train_args_dict, parser, strict=True):
    for k, v in train_args_dict.items():
        if type(v) == tuple:
            parser.add_argument('--' + k, nargs="+", default=v, type=type(v[0]))
        elif type(v) != bool:
            parser.add_argument('--' + k, default=v, type=type(v))
        else:
            parser.add_argument('--' + k, default=int(v), type=int)
    if strict:
        args = parser.parse_args()
    else:
        args, unknown = parser.parse_known_args()
        if unknown:
            print(f"WARNING: ignoring unknown args (residual-only flags injected by run_manager?): {unknown}")
    config = {}
    for key in train_args_dict.keys():
        config[key] = getattr(args, key)
    variant = AttrDict(vars(args))
    variant['train_kwargs'] = config
    return variant, args
