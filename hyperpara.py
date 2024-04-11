
dict_args = {'model_name': 'ssm',
             'tt': None,

             'C': 0.00,
            #  'eval_type': 'nelbo',
             'did':None,
             'gpu':None,
             'combiner_type': 'pog',
             'reg_all': 'all',
             'reg_type': 'l2',
              "dat_seed":-1,
              "cv_seed":-1,
             'otype': 'linear',
             "check_every_n_epoch":1,
             'logger': True,
             'checkpoint_callback': True,
             'gradient_clip_val': 0.0,
             'gradient_clip_algorithm': 'norm',

             'deterministic': False,
             'replace_sampler_ddp': True,

             'prepare_data_per_node': True,

             "optimizer_name":'adam',
             'lr': 0.001,
             'anneal': 1.0,
             'bs': 128,
             'dim_stochastic': 16,
             'nheads': 4,
             'max_epochs': 500,
             'dim_hidden': 64,
             'fold': 1,
             'seed': None
             }
