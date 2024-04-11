
from hyperpara import dict_args
from data.syn_data.create_syn_data import SynData, SynDataIndGene

def generate_dataset_setting():
    # nsamples_syn = 2500
    d_treat = 1
    para_dic = {}
    con = 0
    for n_clusters, d_z, d_x, d_gene in [(5, 5, 8, 8)]:
        # for d_z in [4]:
        # for d_x in [8]:
        for arcoef_trans in [0.5]:
            for nsamples_syn in [1000, 100, 2500, 5000]:
            
                para_dic[con] = {}
                para_dic[con]['n_clusters'] = n_clusters
                para_dic[con]['d_gene'] = d_gene
                para_dic[con]['d_z'] = d_z
                para_dic[con]['d_treat'] = d_treat
                para_dic[con]['d_x'] = d_x
                para_dic[con]['nsamples_syn'] = nsamples_syn
                para_dic[con]['arcoef_trans'] = arcoef_trans

                con += 1
    return para_dic

        
def return_data(dict_args, seed = 1):

    dat_syn = SynData(seed=seed, n_g_group=dict_args['n_clusters'],
                    d_g=dict_args['d_gene'], d_z=dict_args['d_z'],
                    d_u=dict_args['d_treat'], d_x=dict_args['d_x'])
    dat_syn.sample(nsample=dict_args['nsamples_syn'], Tmax=30, v_type="soft", arcoef_trans=dict_args['arcoef_trans'])
    return dat_syn

def return_data2(dict_args, seed = 1):

    dat_syn = SynDataIndGene(seed=seed, n_g_group=dict_args['n_clusters'],
                    d_g=dict_args['d_gene'], d_z=dict_args['d_z'],
                    d_u=dict_args['d_treat'], d_x=dict_args['d_x'])
    dat_syn.sample(nsample=dict_args['nsamples_syn'], Tmax=30, v_type="soft", arcoef_trans=dict_args['arcoef_trans'])
    return dat_syn

