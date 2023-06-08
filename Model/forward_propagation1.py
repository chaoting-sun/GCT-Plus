import torch
from Model.modules import get_src_mask, get_trg_mask


def vaetf_forward_propagation(model, batch, pad_id, use_cond2dec):
    src_mask = get_src_mask(batch['src'], pad_id)
    trg_mask = get_trg_mask(batch['trg'][:, :-1],
                            pad_id, use_cond2dec)

    return model.forward(src=batch['src'],
                         trg=batch['trg'][:, :-1],
                         src_mask=src_mask,
                         trg_mask=trg_mask,
                        )


def scavaetf_forward_propagation(model, batch, pad_id, use_cond2dec):
    src_mask = get_src_mask(batch['src'], pad_id)
    trg_mask = get_trg_mask(batch['trg'][:, :-1], pad_id, use_cond2dec)

    return model.forward(src=batch['src'],
                         trg=batch['trg'][:, :-1],
                         src_mask=src_mask,
                         trg_mask=trg_mask,
                        )


def cvaetf_forward_propagation(model, batch, pad_id, use_cond2dec):
    src_mask = get_src_mask(batch['src'], pad_id, batch['econds'])
    trg_mask = get_trg_mask(batch['trg'][:, :-1], pad_id,
                            use_cond2dec, batch['dconds'])

    return model.forward(src=batch['src'],
                         trg=batch['trg'][:, :-1],
                         src_mask=src_mask,
                         trg_mask=trg_mask,
                         econds=batch['econds'],
                         dconds=batch['dconds'],
                        )
    

def scacvaetfv2_forward_propagation(model, batch, pad_id, use_cond2dec):
    src_enc_mask = get_src_mask(torch.cat((batch['src_scaffold'],
                                           batch['src']), 1),
                                pad_id, batch['econds'])
    # src_enc_mask: (bs, 1, nc+sca+src)
    src_dec_mask = get_src_mask(batch['trg_scaffold'],
                                pad_id, batch['dconds'])
    # src_dec_mask: (bs, 1, nc+<sos>sca<eos>)
    trg_mask = get_trg_mask(batch['trg'][:, :-1], pad_id,
                            batch['dconds'], use_cond2dec)

    return model.forward(src=batch['src'],
                         trg=batch['trg'][:, :-1],
                         src_scaffold=batch['src_scaffold'],
                         trg_scaffold=batch['trg_scaffold'],
                         src_enc_mask=src_enc_mask,
                         src_dec_mask=src_dec_mask,
                         trg_mask=trg_mask,
                         econds=batch['econds'],
                         dconds=batch['dconds'],
                        )


forward_propagation = {
    'vaetf'      : vaetf_forward_propagation,
    'scavaetf'   : scavaetf_forward_propagation,
    'cvaetf'     : cvaetf_forward_propagation,
    'scacvaetfv1': cvaetf_forward_propagation,
    'scacvaetfv2': scacvaetfv2_forward_propagation,
    'scacvaetfv3': cvaetf_forward_propagation,
}


# Handle different model types
# if args.model_type in ('cvaetf', 'scacvaetfv1', 'scacvaetfv3'):
#     src_mask = create_source_mask(
#         batch['src'], args.pad_id, batch['econds'])
#     trg_mask = create_target_mask(
#         trg_input, args.pad_id, batch['dconds'],
#         args.use_cond2dec)

#     # with autocast():
#     preds_prop, preds_mol, mu, log_var, _ = model.forward(
#         src=batch['src'],
#         trg=trg_input,
#         econds=batch['econds'],
#         dconds=batch['dconds'],
#         src_mask=src_mask,
#         trg_mask=trg_mask,
#     )

# elif args.model_type == 'scacvaetfv2':
#     src_enc_mask = create_source_mask(
#         torch.cat((batch['src_scaffold'], batch['src']), 1),
#         args.pad_id, batch['econds'])
#     # src_enc_mask: (bs, 1, nc+sca+src)
#     src_dec_mask = create_source_mask(
#         batch['trg_scaffold'],
#         args.pad_id, batch['dconds'])
#     # src_dec_mask: (bs, 1, nc+<sos>sca<eos>)
#     trg_mask = create_target_mask(
#         trg_input, args.pad_id, batch['dconds'],
#         args.use_cond2dec)

#     # with autocast():
#     preds_prop, preds_mol, mu, log_var, _ = model.forward(
#         src=batch['src'],
#         trg=trg_input,
#         src_scaffold=batch['src_scaffold'],
#         trg_scaffold=batch['trg_scaffold'],
#         econds=batch['econds'],
#         dconds=batch['dconds'],
#         src_enc_mask=src_enc_mask,
#         src_dec_mask=src_dec_mask,
#         trg_mask=trg_mask,
#     )