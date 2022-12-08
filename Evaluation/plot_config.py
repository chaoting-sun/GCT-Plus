
# transformer_dict = {
#     'title': 'Comparison of SNN Start',
#     'figpath': './tf_checkz.png',
#     'sub_file_paths': [f'transformer_ep{ep}' for ep in (21, 22, 23, 24, 25)],
#     'legend_names':   [f'trained tf (ep{ep})' for ep in (21, 22, 23, 24, 25)],
#     'file_name': 'z1z2_statistics.csv'
# }

# aug_all_prefix = 'transformer_ep25_aug-all'
# transformer_aug_all_dict = {
#     'title': 'Comparison of SNN Start (all)',
#     'figpath': './aug-all_tf_checkz.png',
#     'sub_file_paths': [f'transformer_ep{ep}' for ep in (25,)] + 
#                       [f'{aug_all_prefix}_ep{ep}' for ep in (26,27,28,29,30)],
#     'legend_names':   [f'trained tf (ep{ep})' for ep in (25,)] +
#                       [f'aug-all-tf (ep{ep})' for ep in (26,27,28,29,30)],
#     'file_name': 'z1z2_statistics.csv'
# }

# aug_decoderout_prefix = 'transformer_ep25_aug-decoderout'
# transformer_aug_decoderout_dict = {
#     'title': 'Comparison of SNN Start (decoderout)',
#     'figpath': './aug-decoderout_tf_checkconds.png',
#     'sub_file_paths': [f'transformer_ep{ep}' for ep in (25,)] + 
#                       [f'{aug_decoderout_prefix}_ep{ep}' for ep in (26,27,28,29,30)],
#     'legend_names':   [f'trained tf (ep{ep})' for ep in (25,)] +
#                       [f'aug-encoderout-tf (ep{ep})' for ep in (26,27,28,29,30)],
#     'file_name': 'logP_statistics.csv'
# }

# transformer_aug_decoderout_10_dict = {
#     'title': 'Comparison of SNN Start (decoderout)',
#     'figpath': './aug-decoderout_tf_checkconds_10.png',
#     'sub_file_paths': [f'{aug_decoderout_prefix}_ep{ep}' for ep in (26,27,28,29,30,31,32,33,34,35)],
#     'legend_names':   [f'aug-encoderout-tf (ep{ep})' for ep in (26,27,28,29,30,31,32,33,34,35)],
#     'file_name': 'logP_statistics.csv'
# }

# info_dict = {
#     'tf': transformer_dict,
#     'aug-all-tf': transformer_aug_all_dict,
#     'aug-decoderout-tf': transformer_aug_decoderout_dict,
#     'aug-decoderout-10-tf': transformer_aug_decoderout_10_dict
# }


subpath_convert = {
    'transformer': 'transformer',
    'aug_all_tf': 'transformer_ep25_aug-all',
    'aug_encoderout_tf': 'transformer_ep25_aug-decoderout'
}


# transformer_dict = {
#     'title': 'Comparison of SNN Start',
#     'figpath': './tf_checkz.png',
#     'sub_file_paths': [f'transformer_ep{ep}' for ep in (21, 22, 23, 24, 25)],
#     'legend_names':   [f'trained tf (ep{ep})' for ep in (21, 22, 23, 24, 25)],
#     'file_name': 'z1z2_statistics.csv'
# }

def plot_config(query, figname, prop):
    prop_prefix = 'z1z2' if prop == 'z' else prop
    
    sub_file_paths = []
    legend_names = []
    for q in query:
        for ep in q['epoch']:        
            sub_file_paths.append(f"{subpath_convert[q['model']]}_ep{ep}")
            legend_names.append(f"{q['model']} - ep{ep}")
        
    config = {
        'title': 'Comparison of SNN Start',
        'figpath': f"./Evaluation/{figname}",
        'sub_file_paths': sub_file_paths,
        'legend_names': legend_names,
        'file_name': f'{prop_prefix}_statistics.csv',
        'test_for': prop,
    }
    return config


def test_query():
    # model: transformer, aug_all_tf, aug_encoderout_tf
    query1 = { 'model': 'transformer', 'epoch': [25] }
    query2 = { 'model': 'aug_all_tf', 'epoch': [26,27,28,29,30] }
    figname = 'test_query.png'
    return [query1, query2], figname


# query = [query1, quer2, ...], figname
# query_i = { model: ..., epoch: [...] }
# model = transformer | aug_all_tf | aug_encoderout_tf

if __name__ == "__main__":
    query = test_query()
    config = plot_config(query, 'logP')
    print(config)
    