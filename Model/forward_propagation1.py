from Model.modules import get_src_mask, get_trg_mask


def vaetf_forward_propagation(model, batch, pad_id, use_cond2dec):
    src_mask = get_src_mask(batch['src'], pad_id)
    trg_mask = get_trg_mask(batch['trg'][:, :-1],
                            pad_id, use_cond2dec)

    outputs = model.forward(src=batch['src'],
                         trg=batch['trg'][:, :-1],
                         src_mask=src_mask,
                         trg_mask=trg_mask,
                        )
    print(len(outputs))
    return outputs


def scavaetf_forward_propagation(model, batch, pad_id, use_cond2dec):
    src_mask = get_src_mask(batch['src'], pad_id)
    trg_mask = get_trg_mask(batch['trg'][:, :-1], pad_id, use_cond2dec)

    return model.forward(src=batch['src'],
                         trg=batch['trg'][:, :-1],
                         src_mask=src_mask,
                         trg_mask=trg_mask,
                        )


def pvaetf_forward_propagation(model, batch, pad_id, use_cond2dec):
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
    

forward_propagation = {
    'vaetf'    : vaetf_forward_propagation,
    'scavaetf' : scavaetf_forward_propagation,
    'pvaetf'   : pvaetf_forward_propagation,
    'pscavaetf': pvaetf_forward_propagation,
}
