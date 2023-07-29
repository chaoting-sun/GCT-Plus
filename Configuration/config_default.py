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


selected_target_prop = {
    'logP': [ 1.0,   2.0,  3.0],
    'tPSA': [30.0,  60.0, 90.0],
    'QED' : [ 0.6, 0.725, 0.85],
    'SAS' : [ 2.0,  2.75,  3.5],
}


prop_tolerance = { 'logP': 0.4, 'tPSA': 8, 'QED' : 0.03 }

molgpt_selected_target_prop = {
    'logP': [1.0, 3.0],
    'tPSA': [ 40,  80],
    'SAS' : [2.0, 3.5]
}