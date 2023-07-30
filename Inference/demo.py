import numpy as np
import rdkit.Chem as Chem
from Utils.properties import property_prediction


class Demo(object):
    # only use the method in molGCT (decoder + beam search)
    def __init__(self, conditions, predictor, decode_type, latent_dim,
                 max_strlen, use_cond2dec, toklen_data, scaler, TRG,
                 logp_bound, tpsa_bound, qed_bound, decode_algo, device):
        assert len(logp_bound) == 2 and logp_bound[0] <= logp_bound[1]
        assert len(tpsa_bound) == 2 and tpsa_bound[0] <= tpsa_bound[1]
        assert len(qed_bound) == 2 and qed_bound[0] <= qed_bound[1]

        self.decode_type = decode_type
        self.latent_dim = latent_dim
        self.max_strlen = max_strlen
        self.use_cond2dec = use_cond2dec
        self.toklen_data = toklen_data
        self.scaler = scaler
        self.TRG = TRG

        if decode_algo == 'beam_search':
            self.smiles_generator = BeamSearch(predictor, latent_dim, TRG,
                                               toklen_data, scaler, max_strlen,
                                               use_cond2dec, device)
        elif decode_algo == 'multinomial':
            self.smiles_generator = MultinomialSearch(predictor, latent_dim, TRG,
                                                      toklen_data, scaler, max_strlen,
                                                      use_cond2dec, device)
        else:
            exit(f"No decoding algorithm named {decode_algo}")

        self.conditions = conditions
        self.logp_bound = logp_bound
        self.tpsa_bound = tpsa_bound
        self.qed_bound = qed_bound
        
        self.decode_algo = decode_algo
        self.device = device

    def check_properties(self, logp, tpsa, qed):
        assert self.logp_bound[0] <= logp <= self.logp_bound[1]
        assert self.tpsa_bound[0] <= tpsa <= self.tpsa_bound[1]
        assert self.qed_bound[0] <= qed <= self.qed_bound[1]
    
    def inference_from_properties(self, logp, tpsa, qed, 
                                  num_samples, mu=0, std=0.2):
        self.check_properties(logp, tpsa, qed)
        conditions = np.array([[logp, tpsa, qed]])

        if self.decode_type == 'mlp_decode':
            noise = np.random.normal(mu, std, size=(1, len(self.conditions)))
            conditions = (conditions-noise, conditions)

        for i in range(num_samples):
            smiles, toklen_gen, toklen = self.smiles_generator.sample_smiles(conditions)
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                print(f'SMILES {i+1:<6}{smiles:<55} ->\t'
                      f'logP: {property_prediction[self.conditions[0]](mol):.2f}\t'
                      f'tPSA: {property_prediction[self.conditions[1]](mol):.2f}\t'
                      f'QED:  {property_prediction[self.conditions[2]](mol):.2f}')
            else:
                print(f'SMILES {i+1:<6}{smiles:<55} -> X')
    
    def inference_from_src_properties(self, src, logp, tpsa, qed, num_samples, mu, std):
        self.check_properties(logp, tpsa, qed)
        conditions = np.array([[logp, tpsa, qed]])

        if self.decode_type == 'mlp_decode':
            noise = np.random.normal(mu, std, size=(1, len(self.conditions)))
            conditions = (conditions-noise, conditions)

        for i in range(num_samples):
            smiles, toklen_gen, toklen = self.smiles_generator.sample_smiles(src, conditions)
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                print(f'SMILES {i+1:<6}{smiles:<55} ->\t'
                      f'logP: {property_prediction[self.conditions[0]](mol):.2f}\t'
                      f'tPSA: {property_prediction[self.conditions[1]](mol):.2f}\t'
                      f'QED:  {property_prediction[self.conditions[2]](mol):.2f}')
            else:
                print(f'SMILES {i+1:<6}{smiles:<55} -> X')


# def generate_demo(args, model, logp, tpsa, qed, TRG, scaler,
#                   toklen_data, num_samplings, device, mu=0, std=0.2):
    # conditions = np.array([[logp, tpsa, qed]])

#     if args.decode_type == 'mlp_decode':
#         noise = np.random.normal(mu, std, size=(1, args.nconds))
#         conditions = (conditions-noise, conditions)

#     bsTool = BeamSearchTool(args.nconds, args.latent_dim,
#                             args.max_strlen, model, args.use_cond2dec)
#     predictor = ModelPrediction(
#         getattr(model, args.decode_type), args.use_cond2dec)

#     for i in range(num_samplings):
        

#         smiles, _, _ = bsTool.sample_molecule(conditions, toklen_data,
#                                               predictor, TRG, scaler, device)
#         mol = Chem.MolFromSmiles(smiles)
#         if mol is not None:
#             print(f'SMILES {i+1:<6}{smiles:<55} ->\t'
#                   f'logP: {property_prediction[args.conditions[0]](mol):.2f}\t'
#                   f'tPSA: {property_prediction[args.conditions[1]](mol):.2f}\t'
#                   f'QED:  {property_prediction[args.conditions[2]](mol):.2f}')
