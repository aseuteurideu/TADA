import argparse
import pickle, numpy as np
from sklearn.metrics import roc_auc_score
import os
# from utils import min_max_normalize
import sys
# from utils import plot_av_histogram, plot_dist_histogram
import eval_metrics as em
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str,
                    help='Folder location of the test results')
parser.add_argument('--fake_type', default='', type=str, help='Fake type for test')
args = parser.parse_args()

orig_stdout = sys.stdout
f = open(os.path.join(args.folder, args.fake_type + 'output.txt'), 'a')
sys.stdout = f

print('')
print('')
print(args)

def test_process(pred_file='file_pred.pkl'):

    if not os.path.isfile(os.path.join(args.folder, pred_file)):
        return


    # list of size #data x 2. Before softmax.
    with open(os.path.join(args.folder, args.fake_type + pred_file), 'rb') as handle:
        test_pred = pickle.load(handle)

    # list of size #data. 0 or 1
    with open(os.path.join(args.folder, args.fake_type + 'file_target.pkl'), 'rb') as handle:
        test_target = pickle.load(handle)

    # Fix tandem detection cost function (t-DCF) parameters (https://github.com/eurecom-asp/RawGAT-ST-antispoofing/blob/main/tDCF_python/evaluate_tDCF_asvspoof19_eval_LA.py)
    Pspoof = 0.05
    cost_model = {
        'Pspoof': Pspoof,  # Prior probability of a spoofing attack
        'Ptar': (1 - Pspoof) * 0.99,  # Prior probability of target speaker
        'Pnon': (1 - Pspoof) * 0.01,  # Prior probability of nontarget speaker
        'Cmiss_asv': 1,  # Cost of ASV system falsely rejecting target speaker
        'Cfa_asv': 10,  # Cost of ASV system falsely accepting nontarget speaker
        'Cmiss_cm': 1,  # Cost of CM system falsely rejecting target speaker
        'Cfa_cm': 10,  # Cost of CM system falsely accepting spoof
    }

    ############### ACC
    num_correct = 0
    num_data = 0
    for pred, target in zip(test_pred, test_target):
        # acc
        num_data += 1
        pred_index = pred.index(max(pred))
        if pred_index == target:
            num_correct += 1
    print('acc: ', num_correct/num_data*100.0, '%')

    ############### EER (https://github.com/eurecom-asp/RawGAT-ST-antispoofing/blob/main/tDCF_python/evaluate_tDCF_asvspoof19_eval_LA.py#L570

    # Extract bona fide (real bonafide) and spoof scores from the CM scores
    # bona_cm = np.array([pred[1] for i, pred in enumerate(test_pred) if test_target[i]==1])
    # spoof_cm = np.array([pred[1] for i, pred in enumerate(test_pred) if test_target[i]==0])

    test_pred_arr = np.array(test_pred)
    # test_pred_arr = np.array(torch.nn.functional.softmax(torch.tensor(test_pred_arr)))  # doesn't matter. with/without this line, EER and min t-DCF dont change.
    test_target_arr = np.array(test_target)
    bona_cm = test_pred_arr[test_target_arr==1][:,1]
    spoof_cm = test_pred_arr[test_target_arr==0][:,1]

    # EERs of the standalone systems
    eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]

    print('eer: ', eer_cm)

    ################ min t-DCF (https://github.com/eurecom-asp/RawGAT-ST-antispoofing/blob/main/tDCF_python/evaluate_tDCF_asvspoof19_eval_LA.py#L69)
    # Load organizers' ASV scores
    asv_data = np.genfromtxt('ASVspoof2019.LA.asv.eval.gi.trl.scores.txt', dtype=str)
    asv_sources = asv_data[:, 0]
    asv_keys = asv_data[:, 1]
    asv_scores = asv_data[:, 2].astype(float)

    # Extract target, nontarget, and spoof scores from the ASV scores
    tar_asv = asv_scores[asv_keys == 'target']
    non_asv = asv_scores[asv_keys == 'nontarget']
    spoof_asv = asv_scores[asv_keys == 'spoof']

    eer_asv, asv_threshold = em.compute_eer(tar_asv, non_asv)
    [Pfa_asv, Pmiss_asv, Pmiss_spoof_asv] = em.obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold)

    # Compute t-DCF
    tDCF_curve, CM_thresholds = em.compute_tDCF(bona_cm, spoof_cm, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model, True)

    # Minimum t-DCF
    min_tDCF_index = np.argmin(tDCF_curve)
    min_tDCF = tDCF_curve[min_tDCF_index]

    print('min_tDCF: ', min_tDCF)


pred_files = ['file_pred.pkl']
for pf in pred_files:
    print('---------------------------')
    print('prediction file: ', pf)
    test_process(pf)

sys.stdout = orig_stdout
f.close()