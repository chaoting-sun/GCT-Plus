import configuration.opts_mlpcvae as opts
# import configuration.opts as opts

from trainer.mlptransformer_trainer import TransformerTrainer
# from trainer.transformer_trainer import TransformerTrainer

if __name__ == "__main__":
    # opts.train_opts(parser)
    parser = opts.general_opts()
    opt = parser.parse_args()
    # print(' '.join(f'{k}={v}' for k, v in vars(opt).items()))

    trainer = TransformerTrainer(opt)
    trainer.train()