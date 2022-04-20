import configuration.opts as opts
from trainer.transformer_trainer import TransformerTrainer

if __name__ == "__main__":
    # opts.train_opts(parser)
    parser = opts.general_opts()
    opt = parser.parse_args()

    trainer = TransformerTrainer(opt)
    trainer.train()