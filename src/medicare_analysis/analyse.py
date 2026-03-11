import os
import sys
import pandas as pd
import numpy as np
import pickle
import preprocess_files, models
import argparse

parser = argparse.ArgumentParser(description='Generate historic panoids')



# Required positional argument
parser.add_argument('model', type=str,
                    help='model')


DATA_FOLDER = os.path.join('..', 'data')
#
# with open(os.path.join(DATA_FOLDER, 'breg.pickle'), 'rb') as handle:
#     breg = pickle.load(handle)
#
# dct_breg_preds = {'grid_scores':breg['lreg_model'].grid_scores_,
#                   'breg_preds': breg['lreg_preds'],
#                   'n_features': breg['lreg_model'].n_features_
#                   }
#
# with open(os.path.join(DATA_FOLDER, 'breg_preds.pickle'), 'wb') as handle:
#     pickle.dump(dct_breg_preds, handle, protocol=pickle.HIGHEST_PROTOCOL)





dct_filename = {'phy_filename': 'Physician_Compare_National_Downloadable_File.csv',
                'perf_filename': 'Physician_Compare_2015_Individual_EP_Public_Reporting___Performance_Scores.csv',
                'patnt_filename': 'Physician_Compare_2015_Group_Public_Reporting_-_Patient_Experience.csv',
                'claims_filename': 'Medicare_Provider_Utilization_and_Payment_Data__Physician_and_Other_Supplier_PUF_CY2015.csv'
               }

def main(argv):
    args = parser.parse_args()
    model = args.model

    df_claims = preprocess_files.get_data(dct_filename['phy_filename'],
                                          dct_filename['perf_filename'],
                                          dct_filename['patnt_filename'],
                                          dct_filename['claims_filename'])

    with open(os.path.join(DATA_FOLDER, 'cleaned_data.pickle'), 'wb') as handle:
        pickle.dump(df_claims, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if model == 'lreg':

        lreg_model = models.get_lreg_model(df_claims, 0.33, 42)

        with open(os.path.join(DATA_FOLDER, 'lreg.pickle'), 'wb') as handle:
            pickle.dump(lreg_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    elif model =='rforest':

        rforest_model = models.get_rforest_model(df_claims, 0.33, 42)

        with open(os.path.join(DATA_FOLDER, 'rforest_msplit.pickle'), 'wb') as handle:
            pickle.dump(rforest_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    elif model == 'xgboost':

        xgboost_model = models.get_xgboost_model(df_claims, 0.33, 42)

        with open(os.path.join(DATA_FOLDER, 'xgboost_reduced.pickle'), 'wb') as handle:
            pickle.dump(xgboost_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    elif model == 'nnet':

        nnet_model = models.get_nnet_model(df_claims, 0.33, 42)

        with open(os.path.join(DATA_FOLDER, 'nnet_model_1.pickle'), 'wb') as handle:
            pickle.dump(nnet_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    elif model == 'breg':

        lreg_model = models.get_breg_model(df_claims, 0.33, 42)

        with open(os.path.join(DATA_FOLDER, 'breg.pickle'), 'wb') as handle:
            pickle.dump(lreg_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:

        df_physician_with_cpt = models.calculate_code_cluster_diams(df_claims, 'npi', 'hcpcs_code')

        with open(os.path.join(DATA_FOLDER, 'code_cluster_diam.pickle'), 'wb') as handle:
            pickle.dump(df_physician_with_cpt, handle, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    main(sys.argv[1:])