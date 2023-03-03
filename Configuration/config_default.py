"""moses"""

moses_benchmark = {
    'all_property_list': ['logP', 'tPSA', 'QED'],
    'max_strlen': 80,
    'prop_constraints': {
        'logP': [0.03, 4.97],
        'tPSA': [17.92, 112.83],
        'QED' : [0.58, 0.95]
    }
}

"""chembl_02"""

chembl02_benchmark = {
    'all_property_list': ['logP'],
    'max_strlen': 100,
}

benchmark_settings = {
    'moses': moses_benchmark,
    'chembl_02': chembl02_benchmark
}