import os
import pandas as pd
import re
import numpy as np

DATA_FOLDER = os.path.join('..', 'data')

def add_tag_to_score_colnames(id_col, df, scoretag):
    cols = [id_col]
    for col in df.columns:
        if col != id_col:
            cols.append(scoretag + '_' + col)
    df.columns = cols

def pythonize_colnames(df):
    df.columns = list(map(lambda each:re.sub('[^0-9a-zA-Z]+', '_', each).lower(), df.columns))

def get_physician_info(filename):
    df_physician_info = pd.read_csv(os.path.join(DATA_FOLDER, filename))
    df_physician_info = df_physician_info.loc[df_physician_info['State'] == 'IL']
    pythonize_colnames(df_physician_info)
    return df_physician_info

def get_performance_scores(filename):
    df_performance_scores = pd.read_csv(os.path.join(DATA_FOLDER, filename))
    df_performance_scores = df_performance_scores[['NPI', 'Measure Identifier', 'Measure Performance Rate']]
    df_performance_scores.columns = ['npi', 'measure', 'score']
    df_performance_scores = df_performance_scores.pivot(index='npi', columns='measure', values='score').reset_index()

    add_tag_to_score_colnames('npi', df_performance_scores, 'perfrm')

    return df_performance_scores

def get_patient_scores(filename):
    df_patient_scores = pd.read_csv(os.path.join(DATA_FOLDER, filename))
    df_patient_scores = df_patient_scores[['Group PAC ID', 'Measure Identifier',
                                           'Measure Performance Rate']]
    df_patient_scores.columns = ['group_practice_pac_id', 'measure', 'score']
    df_patient_scores = df_patient_scores.pivot(index='group_practice_pac_id', columns='measure',
                                                values='score').reset_index()

    add_tag_to_score_colnames('group_practice_pac_id', df_patient_scores, 'patnt')
    return df_patient_scores

def get_claims(filename):
    df_claims = pd.read_csv(os.path.join(DATA_FOLDER, filename))
    df_claims = df_claims[
        ['National Provider Identifier', 'Gender of the Provider', 'Provider Type', 'Medicare Participation Indicator',
         'Place of Service', 'HCPCS Code', 'HCPCS Drug Indicator', 'Number of Services',
         'Number of Medicare Beneficiaries', 'Number of Distinct Medicare Beneficiary/Per Day Services', 'Average Submitted Charge Amount', 'Average Medicare Payment Amount']]
    df_claims.columns = ['npi', 'sex', 'department', 'is_participant', 'facility_type', 'hcpcs_code', 'includes_drug',
                         'no_service', 'no_medicare_benef', 'no_distinct_medicare_benef_per_day', 'avg_submitted_charge_amt', 'avg_medicare_payment_amt']

    df_claims['overcharge_ratio'] = 1 - df_claims['avg_medicare_payment_amt'] / df_claims['avg_submitted_charge_amt']

    return df_claims



def get_data(phy_filename, perf_filename, patnt_filename, claims_filename):
    df_physician_info = get_physician_info(phy_filename)

    df_patient_scores = get_patient_scores(patnt_filename)

    df_performance_scores = get_performance_scores(perf_filename)

    df_claims = get_claims(claims_filename)

    df_performance_scores = df_performance_scores[
        df_performance_scores['npi'].isin(list(df_physician_info.npi.unique()))]

    df_patient_scores = df_patient_scores[
        df_patient_scores['group_practice_pac_id'].isin(list(df_physician_info['group_practice_pac_id'].unique()))]

    df_physician_info = pd.merge(pd.merge(df_physician_info, df_performance_scores, on='npi', how='left'),
                                 df_patient_scores, on='group_practice_pac_id', how='left')

    df_claims = df_claims[df_claims['npi'].isin(list(df_physician_info['npi'].unique()))]

    df_claims = pd.merge(df_claims, df_physician_info, on='npi', how='left')

    df_claims = df_claims.apply(pd.to_numeric, errors='ignore')

    pythonize_colnames(df_claims)

    df_claims.drop(
        ['pac_id', 'professional_enrollment_id', 'last_name', 'first_name', 'middle_name', 'suffix', 'gender',
         'credential', 'medical_school_name', 'primary_specialty', 'secondary_specialty_1', 'secondary_specialty_2',
         'secondary_specialty_3', 'secondary_specialty_4', 'all_secondary_specialties', 'organization_legal_name',
         'group_practice_pac_id', 'line_1_street_address', 'line_2_street_address',
         'marker_of_address_line_2_suppression', 'city', 'state', 'zip_code', 'phone_number',
         'hospital_affiliation_ccn_1', 'hospital_affiliation_lbn_1', 'hospital_affiliation_ccn_2',
         'hospital_affiliation_lbn_2', 'hospital_affiliation_ccn_3', 'hospital_affiliation_lbn_3',
         'hospital_affiliation_ccn_4', 'hospital_affiliation_lbn_4', 'hospital_affiliation_ccn_5',
         'hospital_affiliation_lbn_5', ], axis=1, inplace=True)

    df_claims = df_claims.dropna(axis=1, how='all')


    return df_claims
