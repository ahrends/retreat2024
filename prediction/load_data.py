import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sb
from scipy.stats import spearmanr
from sklearn.mixture import GaussianMixture
from ppca import PPCA
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV,LogisticRegression,Ridge,RidgeCV,ElasticNetCV,GammaRegressor, HuberRegressor
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
# import regression_things
from datetime import datetime
import sys

data_set = 2 # 0. Bcn; 1. Miriad; 2. Oasis
base_dir = '/Users/au734696/Dropbox/BACKUP/PhD/Data_and_scripts/Python/Retreat'

# some options
modality = 3 # 0: GrayVol, 1: ThickAvg, 2: subcortical, 3: Grayvol+subcortical, 4: ThickAvg+subcortical, 5: SurfArea+subcort
do_zscoring = True
do_zscoring_pcs = False
minT = 3 # minimum of sessions - only reelvant for OASIS
do_robust_regr = False

if data_set == 0:

    # Relevant variables: (no.subjects x variables)
    # Y_cogn_tests: cross-session average and slope of cognitive abilities (1st PC of various tests)
    # Y_apoe and Y_apoe_summary: Genetics
    # Y_csf: cerebrospinal fluid
    # Y_diag: diagnosis
    # Y_cent: centiloid (amyloid)
    # X_mri: MRI information, which depends on the choice of modality.
    #       The first half is cross-session average, second half is slope
    # X_age_gender

    directory = base_dir + '/data/facehbi/'

    mri = np.load(directory + 'mri.npy')
    id = np.load(directory + 'id.npy')
    age = np.load(directory + 'age.npy')
    gender = np.load(directory + 'gender.npy', allow_pickle=True)
    global_cognition = np.load(directory + 'global_cognition.npy')
    diagnosis = np.load(directory + 'diagnosis.npy')
    centiloid = np.load(directory + 'centiloid.npy')

    T = 3
    csv = 'facehbi_long_segs.csv'
    dat = pd.read_csv(directory + csv)

    if do_zscoring: mri = preprocessing.StandardScaler().fit(mri).transform(mri)

    unique_id = np.unique(id)
    N = len(unique_id)

    # choose the subjects with +3 visits and discard 142
    N_final = 0
    choose = np.zeros(mri.shape[0], dtype=bool) 
    choose_id = np.zeros(N, dtype=bool) 
    for j in range(N):
        idj = (id == unique_id[j])
        if int(np.sum(idj))>=3 and (j != 142):
            choose_id[j] = True
            choose[idj] = True
            N_final += 1
    N = N_final

    p = mri.shape[1]

    mri = mri[choose,:]
    id = id[choose]
    age = age[choose]
    gender = gender[choose]
    diagnosis = diagnosis[choose]
    centiloid = centiloid[choose]
    N_scans = np.sum(choose)
    time = np.tile(np.array([-1.0,0.0,1.0]),N)
    unique_id = unique_id[choose_id]

    has_all_centiloids_subj = np.zeros(N, dtype=bool) 
    has_all_centiloids_sess = np.zeros(N_scans, dtype=bool) 
    for j in range(N):
        idj = (id == unique_id[j])
        has_all_centiloids_subj[j] = not np.any(np.isnan(centiloid[idj]))
        has_all_centiloids_sess[idj] = not np.any(np.isnan(centiloid[idj]))

    centiloid[np.logical_not(has_all_centiloids_sess)] = np.nan

    # centiloid_quantile = np.zeros(N_scans)
    # centiloid_quantile[:] = np.nan
    # centiloid_quantile[has_all_centiloids_sess] = \
    #     np.squeeze(
    #         preprocessing.quantile_transform(centiloid[has_all_centiloids_sess].reshape(-1,1),
    #                                      n_quantiles=len(np.unique(centiloid[has_all_centiloids_sess])))
    #     )

    # centiloid_gaussianised = np.zeros(N_scans)
    # centiloid_gaussianised[:] = np.nan
    # centiloid_gaussianised[has_all_centiloids_sess] = \
    #     np.squeeze(
    #         preprocessing.power_transform(centiloid[has_all_centiloids_sess].reshape(-1,1))
    #     )

    # centiloid_sqrt = np.copy(centiloid) 
    # centiloid_sqrt[has_all_centiloids_sess] += 0.1 - np.min(centiloid_sqrt[has_all_centiloids_sess])
    # centiloid_sqrt[has_all_centiloids_sess] = np.sqrt(centiloid_sqrt[has_all_centiloids_sess] )

    # c = centiloid[has_all_centiloids_sess].reshape(-1, 1)
    # gm = GaussianMixture(n_components=2).fit(c)
    # y = gm.fit_predict(c)
    # if np.sum(y==1) > np.sum(y==0): sick = y==0  # 0 is high centiloid
    # else: sick = y==1
    # c[np.logical_not(sick)] = 0.0
    # c[sick] = 5 * c[sick] / np.max(c[sick])
    # centiloid_tweedie = np.zeros(N_scans)
    # centiloid_tweedie[:] = np.nan
    # centiloid_tweedie[has_all_centiloids_sess] = np.squeeze(c)

    # SurfArea,GrayVol,ThickAvg (O. said SurfArea best not)
    cortical_dk_0 = np.empty(0,dtype=int)
    cortical_dk_1 = np.empty(0,dtype=int)
    cortical_dk_2 = np.empty(0,dtype=int)
    subcortical_aseg = np.empty(0,dtype=int)
    wm = np.zeros(p, dtype=bool)

    cortical_dk_dictionary_GrayVol = []
    cortical_dk_dictionary_ThickAvg = []
    cortical_dk_dictionary_SurfArea = []
    subcortical_aseg_dictionary = []

    j_cortical_dk_dictionary_GrayVol = 0
    j_cortical_dk_dictionary_ThickAvg = 0
    j_cortical_dk_dictionary_SurfArea = 0
    j_subcortical_aseg_dictionary = 0

    for j in range(2,len(dat.columns)):
        iregion = j-2 # because we removed Subject_ID and Date earlier
        s = dat.columns[j]
        if s == 'CSF': 
            csf = mri[:,j]
        elif (s == 'eTIV') or (s == 'eTIV.1') or (s == 'Cortex_Thickness') or (s == 'Cortex_Volume'):
            continue
        elif ('WM' in s):
            wm[iregion] = True
        elif ('lh' in s) or ('rh' in s):
            s2 = s[3:]
            dot2 = s2.find('.')
            region = s2[:dot2]
            measure = s2[dot2+1:]
            if measure == 'SurfArea':
                cortical_dk_0 = np.append(cortical_dk_0,np.array([iregion]))
                if s[0:2] == 'lh':
                    cortical_dk_dictionary_SurfArea.append(region + '_left')
                else:
                    cortical_dk_dictionary_SurfArea.append(region + '_right')
                j_cortical_dk_dictionary_SurfArea += 1
            elif measure == 'GrayVol':
                cortical_dk_1 = np.append(cortical_dk_1,np.array([iregion]))
                if s[0:2] == 'lh':
                    # cortical_dk_dictionary_GrayVol[j_cortical_dk_dictionary_GrayVol] = region + '_left'
                    cortical_dk_dictionary_GrayVol.append(region + '_left')
                else:
                    # cortical_dk_dictionary_GrayVol[j_cortical_dk_dictionary_GrayVol] = region + '_right'
                    cortical_dk_dictionary_GrayVol.append(region + '_right')
                j_cortical_dk_dictionary_GrayVol += 1
            elif measure == 'ThickAvg':
                cortical_dk_2 = np.append(cortical_dk_2,np.array([iregion]))
                if s[0:2] == 'lh':
                    # cortical_dk_dictionary_ThickAvg[j_cortical_dk_dictionary_ThickAvg] = region + '_left'
                    cortical_dk_dictionary_ThickAvg.append(region + '_left')
                else:
                    # cortical_dk_dictionary_ThickAvg[j_cortical_dk_dictionary_ThickAvg] = region + '_right'
                    cortical_dk_dictionary_ThickAvg.append(region + '_right')
                j_cortical_dk_dictionary_ThickAvg += 1
        else:
            subcortical_aseg = np.append(subcortical_aseg,np.array([iregion]))
            # subcortical_aseg_dictionary[j_subcortical_aseg_dictionary] = s
            subcortical_aseg_dictionary.append(s)
            j_subcortical_aseg_dictionary += 1


    cort_subcort_0 = np.concatenate((cortical_dk_0,subcortical_aseg))
    cort_subcort_1 = np.concatenate((cortical_dk_1,subcortical_aseg))
    cort_subcort_2 = np.concatenate((cortical_dk_2,subcortical_aseg))

    p_cortical_SurfArea = len(cortical_dk_0)
    p_cortical_GrayVol = len(cortical_dk_1)
    p_cortical_ThickAvg = len(cortical_dk_2)
    p_subcortical = len(subcortical_aseg)
    p_subcort_SurfArea = len(cort_subcort_0)
    p_subcort_GrayVol = len(cort_subcort_1)
    p_subcort_ThickAvg = len(cort_subcort_2)

    if modality == 3:
        indexes_mri_Cort_Subc = np.concatenate((np.ones(p_cortical_GrayVol),2*np.ones(p_subcortical)))
    elif modality == 4:
        indexes_mri_Cort_Subc = np.concatenate((np.ones(p_cortical_ThickAvg),2*np.ones(p_subcortical)))
    elif modality == 5:
        indexes_mri_Cort_Subc = np.concatenate((np.ones(p_cortical_SurfArea),2*np.ones(p_subcortical)))
    
    mri_tensor_cortical_GrayVol = np.zeros((N,T,p_cortical_GrayVol))
    mri_tensor_cortical_ThickAvg = np.zeros((N,T,p_cortical_ThickAvg))
    mri_tensor_cortical_SurfArea = np.zeros((N,T,p_cortical_SurfArea))
    mri_tensor_subcortical =  np.zeros((N,T,p_subcortical))
    mri_tensor_subcort_GrayVol =  np.zeros((N,T,p_subcort_GrayVol))
    mri_tensor_subcort_ThickAvg =  np.zeros((N,T,p_subcort_ThickAvg))
    mri_tensor_subcort_SurfArea =  np.zeros((N,T,p_subcort_SurfArea))

    for j in range(N):

        idj = (id == unique_id[j])
        age_j = age[idj]
        mri_j = mri[idj,:]
        order = np.argsort(age_j)
        mri_j = mri_j[order,:]
        mri_tensor_cortical_SurfArea[j,:,:] = mri_j[:,cortical_dk_0]
        mri_tensor_cortical_GrayVol[j,:,:] = mri_j[:,cortical_dk_1]
        mri_tensor_cortical_ThickAvg[j,:,:] = mri_j[:,cortical_dk_2]
        mri_tensor_subcortical[j,:,:] = mri_j[:,subcortical_aseg]
        mri_tensor_subcort_SurfArea[j,:,:] = mri_j[:,cort_subcort_0]
        mri_tensor_subcort_GrayVol[j,:,:] = mri_j[:,cort_subcort_1]
        mri_tensor_subcort_ThickAvg[j,:,:] = mri_j[:,cort_subcort_2]

    # mri_tensor_cortical_GrayVol_regression = np.zeros((N,2,p_cortical_GrayVol))
    # mri_tensor_cortical_ThickAvg_regression = np.zeros((N,2,p_cortical_ThickAvg))
    # mri_tensor_subcortical_regression =  np.zeros((N,2,p_subcortical))
        
    mri_matrix_cortical_SurfArea = mri[:,cortical_dk_0]
    mri_matrix_cortical_GrayVol = mri[:,cortical_dk_1]
    mri_matrix_cortical_ThickAvg =  mri[:,cortical_dk_2]
    mri_matrix_subcortical =  mri[:,subcortical_aseg]
    mri_matrix_subcort_SurfArea =  mri[:,cort_subcort_0]
    mri_matrix_subcort_GrayVol =  mri[:,cort_subcort_1]
    mri_matrix_subcort_ThickAvg =  mri[:,cort_subcort_2]

    if modality == 0: 
        p = p_cortical_GrayVol
    elif modality == 1: 
        p = p_cortical_ThickAvg  
    elif modality == 2: 
        p = p_subcortical
    elif modality == 3: 
        p = p_subcort_GrayVol
    elif modality == 4: 
        p = p_subcort_ThickAvg
    else: 
        p = p_subcort_SurfArea

    # Cogntitive variables, genetics, CSF

    # do the cognitive variables with prob PCA
    csv = 'cogs.csv'
    dat = pd.read_csv(directory + csv)

    age_cogn = dat.iloc[:,4].values
    id_cogn_tmp = dat.iloc[:,1].values
    id_cogn = np.zeros(len(id_cogn_tmp))
    for j in range(len(id_cogn_tmp)): id_cogn[j] = round(int(id_cogn_tmp[j][1:]))
    cogn_tests = dat.iloc[:,8:14].values

    ppca = PPCA()
    ppca.fit(data=cogn_tests, d=2, verbose=True)
    gi = ppca.transform()
    gi = gi[:,0]
    print(ppca.var_exp)

    Y_cogn_tests = np.zeros((N,2))
    Y_cogn_tests[:] = np.nan
    R = 0.01 * np.eye(2)
    R[0,0] = 0

    for j in range(N):
        idj = id_cogn == unique_id[j]
        age_j = age_cogn[idj]
        gi_j = gi[idj]
        order = np.argsort(age_j)
        age_j = age_j[order]
        x = np.ones((len(age_j),2))
        x[:,1] = age_j - np.mean(age_j)
        y = np.expand_dims(gi_j,1)
        ab = np.linalg.inv(x.T @ x + R) @ (x.T @ y)
        Y_cogn_tests[j,:] = np.squeeze(ab)
        
    # genetics
    apoe_all = dat.iloc[:,-2].values
    Y_apoe = np.zeros((N,5)) # 2/3, 2/4, 3/3, 3/4, and 4/4

    for j in range(N):
        idj = id_cogn == unique_id[j]
        if apoe_all[idj][0] == 'e2e3':
            Y_apoe[j,0] = 1
        elif apoe_all[idj][0] == 'e3e3':
            Y_apoe[j,1] = 1
        elif apoe_all[idj][0] == 'e2e4':
            Y_apoe[j,2] = 1
        elif apoe_all[idj][0] == 'e3e4':
            Y_apoe[j,3] = 1
        elif apoe_all[idj][0] == 'e4e4':
            Y_apoe[j,4] = 1
        else:
            print(apoe_all[idj][0])

    np.sum(Y_apoe,axis=0)

    Y_apoe_summary = np.zeros((N,1))
    # "Association of APOE e2 genotype 
    # with Alzheimer's and non-Alzheimer's neurodegenerative pathologies"
    Y_apoe_summary[Y_apoe[:,2]==1] = 1 
    Y_apoe_summary[Y_apoe[:,3]==1] = 1
    Y_apoe_summary[Y_apoe[:,4]==1] = 1

    print(' ')

    # CSF
    csv = 'csf.csv'
    dat = pd.read_csv(directory + csv)
    # age_csf = dat.iloc[:,4].values
    id_csf_tmp = dat.iloc[:,1].values
    id_csf = np.zeros(len(id_cogn_tmp))
    for j in range(len(id_cogn_tmp)): id_csf[j] = round(int(id_csf_tmp[j][1:]))
    csf = dat.iloc[:,10:12].values

    # n0 = np.zeros(2)
    # n1 = np.zeros(2)
    # n2 = np.zeros(2)
    Y_csf = np.zeros((N,2))
    Y_csf[:] = np.nan
    for j in range(N):
        idj = id_csf == unique_id[j]
        # age_j = age_cogn[idj]
        for i in range(2):
            x = csf[idj,i]
            x = x[np.logical_not(np.isnan(x))]
            # if np.sum(np.logical_not(np.isnan(x))) == 1: 
            #     n1[i] += 1
            # elif np.sum(np.logical_not(np.isnan(x))) == 2:
            #     n2[i] += 1
            # elif np.sum(np.logical_not(np.isnan(x))) == 0:
            #     n0[i] += 1
            if len(x) == 2:
                Y_csf[j,i] = np.mean(x)
            elif len(x) == 1:
                Y_csf[j,i] = x[0]

    # Diagnosis : any diagnosis of MCI?

    diagnosis_subj = np.zeros(N)

    for i in range(N):
        ii = (id == unique_id[i]).nonzero()[0]
        diagnosis_subj[i] = np.sum(diagnosis[ii])

    Y_diag = np.zeros((N,1),dtype=int) 
    for j in range(N):
        idj = (id == unique_id[j])
        diagnosis_j = diagnosis[idj]
        Y_diag[j] = np.any(diagnosis_j == 1)

    # Centiloid

    Y_cent = np.zeros((N,2),dtype=float) 
    Y_cent[:] = np.nan

    for j in range(N):
        idj = (id == unique_id[j])
        age_j = age[idj]
        order = np.argsort(age_j)
        age_j = age_j[order]
        centiloid_j = centiloid[idj]
        centiloid_j = centiloid_j[order]   
        if has_all_centiloids_subj[j]:
            x = np.ones((3,2))
            x[:,1] = age_j - np.mean(age_j)
            y = np.expand_dims(centiloid_j,1)
            ab = np.linalg.inv(x.T @ x + R) @ (x.T @ y)
            Y_cent[j,:] = np.squeeze(ab)

    for j in range(Y_cent.shape[1]):
        not_nan_test = np.logical_not(np.isnan(Y_cent[:,j]))
        Y_cent[not_nan_test,j] -= np.min(Y_cent[not_nan_test,j],axis=0)
        Y_cent[not_nan_test,j] += 0.01

    Y = np.concatenate((Y_cogn_tests,Y_apoe_summary,Y_csf,Y_cent), axis=1)

    dist_gamma = np.zeros(Y.shape[1])
    dist_gamma[-2:] = 1

    # Build X for prediction, subject data for MRI and clinical stuff respectively

    R = 0.1 * np.eye(2)
    R[0,0] = 0
    A = np.zeros((N,p))
    B = np.zeros((N,p))

    for i in range(N):
        ii = (id == unique_id[i]).nonzero()[0]
        age_i = age[ii]
        order = np.argsort(age_i)
        age_i = age_i[order]
        # regression
        x = np.ones((3,2))
        x[:,1] = age_i - np.mean(age_i)
        for j in range(p):
            if modality == 0: y = mri_matrix_cortical_GrayVol[ii,j]
            elif modality == 1: y = mri_matrix_cortical_ThickAvg[ii,j]
            elif modality == 2: y = mri_matrix_subcortical[ii,j]
            elif modality == 3: y = mri_matrix_subcort_GrayVol[ii,j]
            elif modality == 4: y = mri_matrix_subcort_ThickAvg[ii,j]
            elif modality == 5: y = mri_matrix_subcort_SurfArea[ii,j]
            y = np.expand_dims(y[order],1)
            ab = np.linalg.inv(x.T @ x + R) @ (x.T @ y)
            A[i,j] = ab[0,0]
            B[i,j] = ab[1,0]

    # defined above from a regression on each ROI
    scaler = preprocessing.StandardScaler().fit(B)
    B = scaler.transform(B)
    scaler = preprocessing.StandardScaler().fit(A)
    A = scaler.transform(A)

    X_mri = np.concatenate((A,B), axis=1)

    X_age_gender = np.zeros((N,2))
    for i in range(N):
        ii = (id == unique_id[i]).nonzero()[0]
        age_i = age[ii]
        order = np.argsort(age_i)
        age_i = age_i[order]
        gender_i = gender[ii]
        if len(np.unique(gender_i))>1: raise Exception('shit')
        X_age_gender[i,0] = np.mean(age_i)   
        if gender_i[0] == 'female': X_age_gender[i,1] = -1
        else: X_age_gender[i,1] = +1

    scaler = preprocessing.StandardScaler().fit(X_age_gender)
    X_age_gender = scaler.transform(X_age_gender)


elif data_set == 1:

    # Relevant variables: (no.subjects x variables)
    # Y_cogn: cross-session average and slope of cognitive abilities (1st PC of various tests)
    # Y_apoe: Genetics
    # Y_diag: diagnosis
    # X_mri: MRI information, which depends on the choice of modality.
    #       The first half is cross-session average, second half is slope
    # X_age_gender

    directory = base_dir + '/data/miriad/'

    csv = 'all_meas.txt'
    dat = pd.read_csv(directory + csv)
    mri = dat.iloc[:,2:].values
    id = dat.iloc[:,0].values
    age = dat.iloc[:,1].values

    unique_id = np.unique(id)
    N = len(unique_id)

    if do_zscoring: mri = preprocessing.StandardScaler().fit(mri).transform(mri)

    p = mri.shape[1]

    # SurfArea,GrayVol,ThickAvg (O. said SurfArea best not)
    cortical_dk_0 = np.empty(0,dtype=int)
    cortical_dk_1 = np.empty(0,dtype=int)
    cortical_dk_2 = np.empty(0,dtype=int)
    subcortical_aseg = np.empty(0,dtype=int)
    wm = np.zeros(p, dtype=bool)

    cortical_dk_dictionary_GrayVol = []
    cortical_dk_dictionary_ThickAvg = []
    cortical_dk_dictionary_SurfArea = []
    subcortical_aseg_dictionary = []

    j_cortical_dk_dictionary_GrayVol = 0
    j_cortical_dk_dictionary_ThickAvg = 0
    j_cortical_dk_dictionary_SurfArea = 0
    j_subcortical_aseg_dictionary = 0

    for j in range(2,len(dat.columns)):
        iregion = j-2 # because we removed Subject_ID and Date earlier
        s = dat.columns[j]
        if s == 'CSF': 
            csf = mri[:,j]
        elif (s == 'eTIV') or (s == 'eTIV.1') or (s == 'Cortex_Thickness') or (s == 'Cortex_Volume'):
            continue
        elif ('WM' in s):
            wm[iregion] = True
        elif ('lh' in s) or ('rh' in s):
            s2 = s[3:]
            dot2 = s2.find('.')
            region = s2[:dot2]
            measure = s2[dot2+1:]
            if measure == 'SurfArea':
                cortical_dk_0 = np.append(cortical_dk_0,np.array([iregion]))
                if s[0:2] == 'lh':
                    cortical_dk_dictionary_SurfArea.append(region + '_left')
                else:
                    cortical_dk_dictionary_SurfArea.append(region + '_right')
                j_cortical_dk_dictionary_SurfArea += 1
            elif measure == 'GrayVol':
                cortical_dk_1 = np.append(cortical_dk_1,np.array([iregion]))
                if s[0:2] == 'lh':
                    # cortical_dk_dictionary_GrayVol[j_cortical_dk_dictionary_GrayVol] = region + '_left'
                    cortical_dk_dictionary_GrayVol.append(region + '_left')
                else:
                    # cortical_dk_dictionary_GrayVol[j_cortical_dk_dictionary_GrayVol] = region + '_right'
                    cortical_dk_dictionary_GrayVol.append(region + '_right')
                j_cortical_dk_dictionary_GrayVol += 1
            elif measure == 'ThickAvg':
                cortical_dk_2 = np.append(cortical_dk_2,np.array([iregion]))
                if s[0:2] == 'lh':
                    # cortical_dk_dictionary_ThickAvg[j_cortical_dk_dictionary_ThickAvg] = region + '_left'
                    cortical_dk_dictionary_ThickAvg.append(region + '_left')
                else:
                    # cortical_dk_dictionary_ThickAvg[j_cortical_dk_dictionary_ThickAvg] = region + '_right'
                    cortical_dk_dictionary_ThickAvg.append(region + '_right')
                j_cortical_dk_dictionary_ThickAvg += 1
        else:
            subcortical_aseg = np.append(subcortical_aseg,np.array([iregion]))
            # subcortical_aseg_dictionary[j_subcortical_aseg_dictionary] = s
            subcortical_aseg_dictionary.append(s)
            j_subcortical_aseg_dictionary += 1


    cort_subcort_0 = np.concatenate((cortical_dk_0,subcortical_aseg))
    cort_subcort_1 = np.concatenate((cortical_dk_1,subcortical_aseg))
    cort_subcort_2 = np.concatenate((cortical_dk_2,subcortical_aseg))

    p_cortical_SurfArea = len(cortical_dk_0)
    p_cortical_GrayVol = len(cortical_dk_1)
    p_cortical_ThickAvg = len(cortical_dk_2)
    p_subcortical = len(subcortical_aseg)
    p_subcort_SurfArea = len(cort_subcort_0)
    p_subcort_GrayVol = len(cort_subcort_1)
    p_subcort_ThickAvg = len(cort_subcort_2)

    if modality == 3:
        indexes_mri_Cort_Subc = np.concatenate((np.ones(p_cortical_GrayVol),2*np.ones(p_subcortical)))
    elif modality == 4:
        indexes_mri_Cort_Subc = np.concatenate((np.ones(p_cortical_ThickAvg),2*np.ones(p_subcortical)))
    elif modality == 5:
        indexes_mri_Cort_Subc = np.concatenate((np.ones(p_cortical_SurfArea),2*np.ones(p_subcortical)))

    mri_matrix_cortical_SurfArea_miriad = mri[:,cortical_dk_0]
    mri_matrix_cortical_GrayVol_miriad = mri[:,cortical_dk_1]
    mri_matrix_cortical_ThickAvg_miriad =  mri[:,cortical_dk_2]
    mri_matrix_subcortical_miriad =  mri[:,subcortical_aseg]

    mri_matrix_subcort_SurfArea_miriad =  mri[:,cort_subcort_0]
    mri_matrix_subcort_GrayVol_miriad =  mri[:,cort_subcort_1]
    mri_matrix_subcort_ThickAvg_miriad =  mri[:,cort_subcort_2]

    if modality == 0: 
        p = p_cortical_GrayVol
    elif modality == 1: 
        p = p_cortical_ThickAvg  
    elif modality == 2: 
        p = p_subcortical
    elif modality == 3: 
        p = p_subcort_GrayVol
    elif modality == 4: 
        p = p_subcort_ThickAvg
    else: 
        p = p_subcort_SurfArea

    # Build X for prediction

    R = 0.1 * np.eye(2)
    R[0,0] = 0
    A = np.zeros((N,p))
    B = np.zeros((N,p))

    for i in range(N):
        ii = (id == unique_id[i]).nonzero()[0]
        age_i = age[ii]
        order = np.argsort(age_i)
        age_i = age_i[order]
        T = len(age_i)
        # regression
        x = np.ones((T,2))
        x[:,1] = age_i - np.mean(age_i)
        for j in range(p):
            if modality == 3: y =  mri_matrix_subcort_GrayVol_miriad[ii,j]
            elif modality == 4: y = mri_matrix_subcort_ThickAvg_miriad[ii,j]
            elif modality == 5: y = mri_matrix_subcort_SurfArea_miriad[ii,j]
            if do_robust_regr:
                y = y[order]
                model = HuberRegressor().fit(np.expand_dims(x[:,1],1), y)
                A[i,j] = model.intercept_
                B[i,j] = model.coef_[0]
            else:
                y = np.expand_dims(y[order],1)
                ab = np.linalg.inv(x.T @ x + R) @ (x.T @ y)
                A[i,j] = ab[0,0]
                B[i,j] = ab[1,0]

    # defined above from a regression on each ROI
    scaler = preprocessing.StandardScaler().fit(B)
    B = scaler.transform(B)
    scaler = preprocessing.StandardScaler().fit(A)
    A = scaler.transform(A)

    X_mri = np.concatenate((A,B), axis=1)

    # Collect behavioural performance, age and gender
    csv = 'behavioural_scores.txt'
    dat = pd.read_csv(directory + csv)

    id_c = dat.iloc[:,0].values
    age_c = dat.iloc[:,1].values
    gender = dat.iloc[:,2].values
    unique_id_c = np.unique(id)

    X_age_gender = np.zeros((N,2))
    for i in range(N):
        id1 = unique_id[i]
        ii = (id == id1).nonzero()[0]
        age_i = age[ii]
        order = np.argsort(age_i)
        age_i = age_i[order]
        id2 = unique_id_c[i]
        ii = (id_c == id2).nonzero()[0]
        gender_i = gender[ii]
        if len(np.unique(gender_i))>1: raise Exception('shit')
        X_age_gender[i,0] = np.mean(age_i)   
        if gender_i[0] == 'female': X_age_gender[i,1] = -1
        else: X_age_gender[i,1] = +1

    scaler = preprocessing.StandardScaler().fit(X_age_gender)
    X_age_gender = scaler.transform(X_age_gender)

    mmse = dat.iloc[:,-1].values
    mmse = -np.log(mmse)
    mmse -= np.min(mmse) 
    mmse += 0.1
    # plt.hist(mmse)

    A_c = np.zeros((N,1))
    A_c[:] = np.nan
    B_c = np.zeros((N,1))
    B_c[:] = np.nan

    for i in range(N):
        id1 = unique_id[i]
        ii = (id_c == id1).nonzero()[0] # unique_id_c[i]
        age_i = age_c[ii]
        y = mmse[ii]
        T = len(np.unique(age_i))
        if T>2 and not np.any(np.isnan(age_i)):
            order = np.argsort(age_i)
            age_i = age_i[order]
            # regression
            x = np.ones((T,2))
            x[:,1] = age_i - np.mean(age_i)
            y = np.expand_dims(y[order],1)
            ab = np.linalg.inv(x.T @ x + R) @ (x.T @ y)
            A_c[i,0] = ab[0,0]
            B_c[i,0] = ab[1,0]    
        else:
            A_c[i,0] = np.mean(y)

    Y_cogn = np.concatenate((A_c,B_c),axis=1)
    for j in range(2):
        not_nan_test = np.logical_not(np.isnan(Y_cogn[:,j]))
        Y_cogn[not_nan_test,j] -= np.min(Y_cogn[not_nan_test,j]) - 0.01

    dist_gamma = np.zeros(5)
    dist_gamma[2:4] = 1

    # APOE
    csv = 'apoe.tsv'
    dat = pd.read_csv(directory + csv,sep='\t')
    id_tmp = dat.iloc[:,0].values
    id_a = np.zeros(N)
    for j in range(N): id_a[j] = round(int(id_tmp[j][4:]))
    apoe_all = dat.iloc[:,1].values
    Y_apoe = np.zeros((N,5)) # 2/3, 2/4, 3/3, 3/4, and 4/4

    for j in range(N):
        idj = (id_a == unique_id[j]).nonzero()[0]
        if apoe_all[idj] == 23:
            Y_apoe[j,0] = 1
        elif apoe_all[idj] == 33:
            Y_apoe[j,1] = 1
        elif apoe_all[idj] == 24:
            Y_apoe[j,2] = 1
        elif apoe_all[idj] == 34:
            Y_apoe[j,3] = 1
        elif apoe_all[idj] == 44:
            Y_apoe[j,4] = 1

    Y_apoe_summary = np.zeros((N,1))
    # "Association of APOE e2 genotype 
    # with Alzheimer's and non-Alzheimer's neurodegenerative pathologies"
    Y_apoe_summary[Y_apoe[:,2]==1] = 1 
    Y_apoe_summary[Y_apoe[:,3]==1] = 1
    Y_apoe_summary[Y_apoe[:,4]==1] = 1

    # Diagnosis 

    csv = 'lenrui_5_4_2023_15_57_14.csv'
    dat = pd.read_csv(directory + csv)
    id_tmp = dat.iloc[:,0].values
    id_d = np.zeros(len(id_tmp))
    for j in range(len(id_tmp)): id_d[j] = round(int(id_tmp[j][7:10]))
    diagnosis = dat.iloc[:,3].values

    # diagnosis_subj = np.zeros((N,2))
    Y_diag = np.zeros((N,1),dtype=int) 
    for i in range(N):
        ii = (id_d == unique_id[i]).nonzero()[0]
        Y_diag[i] = int(np.sum(diagnosis[ii] == 'AD') > 0)
        # diagnosis_subj[i,0] = np.sum(diagnosis[ii] == 'AD')
        # diagnosis_subj[i,1] = np.sum(diagnosis[ii] == 'Control')


else: 

    # Relevant variables: (no.subjects x variables)
    # Y_mmse: cross-session average and slope of minimental (test for diagnosis of AD)
    # Y_cdr: Clinical Dementia Rating
    # Y_sumbox: A more detailed thing than the CDR, 
    #           with a range of 0 (no impairment) to 18 (severe impairment)
    # Y_diag: diagnosis
    # Y_apoe, Y_apoe_summary: Genetics
    # X_mri: MRI information, which depends on the choice of modality.
    #       The first half is cross-session average, second half is slope
    # X_age_gender

    directory = base_dir + '/data/oasis/'

    csv = 'all_meas.txt'
    dat = pd.read_csv(directory + csv)
    mri = dat.iloc[:,2:].values
    id = dat.iloc[:,0].values
    age = dat.iloc[:,1].values

    unique_id = np.unique(id)
    N = len(unique_id)

    if do_zscoring: mri = preprocessing.StandardScaler().fit(mri).transform(mri)

    # choose the subjects with +maxT visits
    N_final = 0
    choose = np.zeros(mri.shape[0], dtype=bool) 
    choose_id = np.zeros(N, dtype=bool) 
    T = np.zeros(N)
    for j in range(N):
        idj = (id == unique_id[j])
        T[j] = np.sum(idj)
        if T[j] >= minT:
            choose_id[j] = True
            choose[idj] = True
            N_final += 1
    N = N_final

    p = mri.shape[1]
    id = id[choose]
    age = age[choose]
    mri = mri[choose,:]
    N_scans = np.sum(choose)
    unique_id = unique_id[choose_id]

    # SurfArea,GrayVol,ThickAvg (O. said SurfArea best not)
    cortical_dk_0 = np.empty(0,dtype=int)
    cortical_dk_1 = np.empty(0,dtype=int)
    cortical_dk_2 = np.empty(0,dtype=int)
    subcortical_aseg = np.empty(0,dtype=int)
    wm = np.zeros(p, dtype=bool)

    cortical_dk_dictionary_GrayVol = []
    cortical_dk_dictionary_ThickAvg = []
    cortical_dk_dictionary_SurfArea = []
    subcortical_aseg_dictionary = []

    j_cortical_dk_dictionary_GrayVol = 0
    j_cortical_dk_dictionary_ThickAvg = 0
    j_cortical_dk_dictionary_SurfArea = 0
    j_subcortical_aseg_dictionary = 0

    for j in range(2,len(dat.columns)):
        iregion = j-2 # because we removed Subject_ID and Date earlier
        s = dat.columns[j]
        if s == 'CSF': 
            csf = mri[:,j]
        elif (s == 'eTIV') or (s == 'eTIV.1') or (s == 'Cortex_Thickness') or (s == 'Cortex_Volume'):
            continue
        elif ('WM' in s):
            wm[iregion] = True
        elif ('lh' in s) or ('rh' in s):
            s2 = s[3:]
            dot2 = s2.find('.')
            region = s2[:dot2]
            measure = s2[dot2+1:]
            if measure == 'SurfArea':
                cortical_dk_0 = np.append(cortical_dk_0,np.array([iregion]))
                if s[0:2] == 'lh':
                    cortical_dk_dictionary_SurfArea.append(region + '_left')
                else:
                    cortical_dk_dictionary_SurfArea.append(region + '_right')
                j_cortical_dk_dictionary_SurfArea += 1
            elif measure == 'GrayVol':
                cortical_dk_1 = np.append(cortical_dk_1,np.array([iregion]))
                if s[0:2] == 'lh':
                    # cortical_dk_dictionary_GrayVol[j_cortical_dk_dictionary_GrayVol] = region + '_left'
                    cortical_dk_dictionary_GrayVol.append(region + '_left')
                else:
                    # cortical_dk_dictionary_GrayVol[j_cortical_dk_dictionary_GrayVol] = region + '_right'
                    cortical_dk_dictionary_GrayVol.append(region + '_right')
                j_cortical_dk_dictionary_GrayVol += 1
            elif measure == 'ThickAvg':
                cortical_dk_2 = np.append(cortical_dk_2,np.array([iregion]))
                if s[0:2] == 'lh':
                    # cortical_dk_dictionary_ThickAvg[j_cortical_dk_dictionary_ThickAvg] = region + '_left'
                    cortical_dk_dictionary_ThickAvg.append(region + '_left')
                else:
                    # cortical_dk_dictionary_ThickAvg[j_cortical_dk_dictionary_ThickAvg] = region + '_right'
                    cortical_dk_dictionary_ThickAvg.append(region + '_right')
                j_cortical_dk_dictionary_ThickAvg += 1
        else:
            subcortical_aseg = np.append(subcortical_aseg,np.array([iregion]))
            # subcortical_aseg_dictionary[j_subcortical_aseg_dictionary] = s
            subcortical_aseg_dictionary.append(s)
            j_subcortical_aseg_dictionary += 1


    cort_subcort_0 = np.concatenate((cortical_dk_0,subcortical_aseg))
    cort_subcort_1 = np.concatenate((cortical_dk_1,subcortical_aseg))
    cort_subcort_2 = np.concatenate((cortical_dk_2,subcortical_aseg))

    p_cortical_SurfArea = len(cortical_dk_0)
    p_cortical_GrayVol = len(cortical_dk_1)
    p_cortical_ThickAvg = len(cortical_dk_2)
    p_subcortical = len(subcortical_aseg)
    p_subcort_SurfArea = len(cort_subcort_0)
    p_subcort_GrayVol = len(cort_subcort_1)
    p_subcort_ThickAvg = len(cort_subcort_2)

    if modality == 3:
        indexes_mri_Cort_Subc = np.concatenate((np.ones(p_cortical_GrayVol),2*np.ones(p_subcortical)))
    elif modality == 4:
        indexes_mri_Cort_Subc = np.concatenate((np.ones(p_cortical_ThickAvg),2*np.ones(p_subcortical)))
    elif modality == 5:
        indexes_mri_Cort_Subc = np.concatenate((np.ones(p_cortical_SurfArea),2*np.ones(p_subcortical)))

    mri_matrix_cortical_SurfArea_oasis = mri[:,cortical_dk_0]
    mri_matrix_cortical_GrayVol_oasis = mri[:,cortical_dk_1]
    mri_matrix_cortical_ThickAvg_oasis =  mri[:,cortical_dk_2]
    mri_matrix_subcortical_oasis =  mri[:,subcortical_aseg]

    mri_matrix_subcort_SurfArea_oasis =  mri[:,cort_subcort_0]
    mri_matrix_subcort_GrayVol_oasis =  mri[:,cort_subcort_1]
    mri_matrix_subcort_ThickAvg_oasis =  mri[:,cort_subcort_2]

    if modality == 0: 
        p = p_cortical_GrayVol
    elif modality == 1: 
        p = p_cortical_ThickAvg  
    elif modality == 2: 
        p = p_subcortical
    elif modality == 3: 
        p = p_subcort_GrayVol
    elif modality == 4: 
        p = p_subcort_ThickAvg
    else: 
        p = p_subcort_SurfArea


    # Build X for prediction

    R = 0.1 * np.eye(2)
    R[0,0] = 0
    A = np.zeros((N,p))
    B = np.zeros((N,p))
    age_mri = np.zeros((N,1))

    for i in range(N):
        ii = (id == unique_id[i]).nonzero()[0]
        age_i = age[ii]
        not_nan = np.logical_not(np.isnan(age_i))
        age_i = age_i[not_nan]
        order = np.argsort(age_i)
        age_i = age_i[order]
        age_mri[i] = np.mean(age_i)
        T = len(age_i)
        # regression
        x = np.ones((T,2))
        x[:,1] = age_i - np.mean(age_i)
        for j in range(p):
            if modality == 3: y =   mri_matrix_subcort_GrayVol_oasis[ii,j]
            elif modality == 4: y = mri_matrix_subcort_ThickAvg_oasis[ii,j]
            elif modality == 5: y = mri_matrix_subcort_SurfArea_oasis[ii,j]
            y = np.expand_dims(y[order],1)
            ab = np.linalg.inv(x.T @ x + R) @ (x.T @ y)
            A[i,j] = ab[0,0]
            B[i,j] = ab[1,0]

    # defined above from a regression on each ROI
    scaler = preprocessing.StandardScaler().fit(B)
    B = scaler.transform(B)
    scaler = preprocessing.StandardScaler().fit(A)
    A = scaler.transform(A)

    X_mri = np.concatenate((A,B), axis=1)

    # Collect behavioural performance, age and gender
    csv = 'clin_ass.txt'
    dat = pd.read_csv(directory + csv)
    id_c = dat.iloc[:,0].values
    age_c = dat.iloc[:,1].values
    cdr = dat.iloc[:,2].values # clinical dementia score
    gender_c = dat.iloc[:,-3].values
    race = dat.iloc[:,-2].values
    mmse = dat.iloc[:,-1].values
    apoe = dat.iloc[:,14].values
    sumbox = dat.iloc[:,15].values # another clinical dementia score, with a range of 0 (no impairment) to 18 (severe impairment)

    unique_id_c = np.unique(id_c)
    N_c = len(unique_id_c)
    T_c = np.zeros(N_c)

    choose_id = np.zeros(N, dtype=bool)
    choose_c = np.ones(id_c.shape[0], dtype=bool) 
    choose_id_c = np.ones(N_c, dtype=bool) 

    T_c = np.zeros(N_c)
    for j in range(N_c):
        idj = (id_c == unique_id_c[j])
        T_c[j] = np.sum(idj)
        if T_c[j] < minT: # or (not np.any(unique_id == unique_id_c[j]))
            choose_id_c[j] = False
            choose_c[idj] = False
        else:
            i = (unique_id == unique_id_c[j]).nonzero()[0]
            choose_id[i] = True

    age_mri = age_mri[choose_id,:]
    X_mri = X_mri[choose_id,:]
    unique_id = unique_id[choose_id]
    unique_id_c = unique_id_c[choose_id_c]

    N = np.sum(choose_id)

    id_c = id_c[choose_c]
    age_c = age_c[choose_c]
    cdr = cdr[choose_c]
    gender_c = gender_c[choose_c]
    race = race[choose_c]
    mmse = mmse[choose_c]
    apoe = apoe[choose_c]
    sumbox = sumbox[choose_c]

    # mmse = -np.log(mmse)
    mmse = -mmse
    mmse -= np.min(mmse[np.logical_not(np.isnan(mmse))]) 
    mmse += 0.1

    # Diagnostic accuracy of the Clinical Dementia Rating Scale for detecting mild cognitive impairment and dementia: A bivariate meta‐analysis
    # The CDR‐GS represents the severity of dementia based on a scale of 
    # 0–3 with “0” indicating nodementia, 
    # “0.5” indicating questionable dementia (MCI),
    # “1” indicating mild dementia,
    # “2” indicating moderate dementia,
    # and “3” indicating severe dementia.
    # The CDR‐SB summarizes each of the domain box scores , with scores ranging from 0 (no dementia) to 18 (severe dementia) 
    # and a higher score indicating more severe functional and cognitive impairment

    X_age_gender = np.zeros((N,2))
    Y_diag = np.zeros((N,3),dtype=int) # healthy, MCI, AD

    R = 0.1 * np.eye(2)
    R[0,0] = 0
    mmse_A = np.zeros((N,1))
    mmse_A[:] = np.nan
    mmse_B = np.zeros((N,1))
    mmse_B[:] = np.nan
    sumbox_A = np.zeros((N,1))
    sumbox_A[:] = np.nan
    sumbox_B = np.zeros((N,1))
    sumbox_B[:] = np.nan

    Y_apoe = np.zeros((N,6))
    Y_cdr = np.zeros((N,2))

    for i in range(N):
        ii = (id_c == unique_id[i]).nonzero()[0]
        age_i = age_c[ii]
        # not_nan = np.logical_not(np.isnan(age_i))
        # age_i = age_i[not_nan]
        # if np.any(cdr[ii] != cdr[ii][0]):
        #     raise Exception('woops')
        if np.nanmax(cdr[ii]) == 0:
            Y_diag[i,0] = 1
        elif (np.nanmax(cdr[ii]) >= 1):
            Y_diag[i,2] = 1
        else:
            Y_diag[i,1] = 1
        X_age_gender[i,0] = np.nanmean(age_i) 
        if gender_c[ii][0] == 'F': X_age_gender[i,1] = -1
        else: X_age_gender[i,1] = +1
        # Y_cdr
        Y_cdr[i,0] = np.nanmax(cdr[ii])
        Y_cdr[i,1] = np.nanmean(cdr[ii])
        # mmse 
        age_i = age_c[ii]
        mmse_i = mmse[ii]
        is_not_nan = np.logical_and(np.logical_not(np.isnan(mmse_i)),np.logical_not(np.isnan(age_i)))
        mmse_i = mmse_i[is_not_nan]
        age_i = age_i[is_not_nan]
        order = np.argsort(age_i)
        age_i = age_i[order]
        mmse_i = mmse_i[order]
        y = mmse_i
        x = np.ones((mmse_i.shape[0],2))
        x[:,1] = age_i - np.nanmean(age_i)
        y = np.expand_dims(y[order],1)
        ab = np.linalg.inv(x.T @ x + R) @ (x.T @ y)
        mmse_A[i,0] = ab[0,0]
        mmse_B[i,0] = ab[1,0]  
        # sumbox
        age_i = age_c[ii]
        sumbox_i = sumbox[ii]
        is_not_nan = np.logical_and(np.logical_not(np.isnan(sumbox_i)),np.logical_not(np.isnan(age_i)))
        sumbox_i = sumbox_i[is_not_nan]
        age_i = age_i[is_not_nan]
        order = np.argsort(age_i)
        age_i = age_i[order]
        sumbox_i = sumbox_i[order]
        y = sumbox_i
        x = np.ones((sumbox_i.shape[0],2))
        x[:,1] = age_i - np.mean(age_i)
        y = np.expand_dims(y[order],1)
        ab = np.linalg.inv(x.T @ x + R) @ (x.T @ y)
        sumbox_A[i,0] = ab[0,0]
        sumbox_B[i,0] = ab[1,0]  
        # apoe
        if np.any(apoe[ii] != apoe[ii][0]):
            raise Exception('woops')   
        if apoe[ii][0] == 22:
            Y_apoe[i,0] = 1
        elif apoe[ii][0] == 23:
            Y_apoe[i,1] = 1
        elif apoe[ii][0] == 33:
            Y_apoe[i,2] = 1
        elif apoe[ii][0] == 24:
            Y_apoe[i,3] = 1
        elif apoe[ii][0] == 34:
            Y_apoe[i,4] = 1
        elif apoe[ii][0] == 44:
            Y_apoe[i,5] = 1
            

    Y_mmse = np.concatenate((mmse_A,mmse_B),axis=1)
    Y_sumbox = np.concatenate((sumbox_A,sumbox_B),axis=1)

    Y_apoe_summary = np.zeros((N,1))
    # "Association of APOE e2 genotype 
    # with Alzheimer's and non-Alzheimer's neurodegenerative pathologies"
    Y_apoe_summary[Y_apoe[:,3]==1] = 1 
    Y_apoe_summary[Y_apoe[:,4]==1] = 1
    Y_apoe_summary[Y_apoe[:,5]==1] = 1

    Y = np.concatenate((Y_cdr,Y_mmse,Y_sumbox,Y_apoe_summary),axis=1)
    Y -= np.min(Y,axis=0)
    Y += 0.01

    Y_strat = np.zeros(N)
    Y_strat[(Y_diag[:,1] + Y_diag[:,2]) > 0] = 1

