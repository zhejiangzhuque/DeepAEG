import random, os, sys
import numpy as np
import csv
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.optimizers import Adam
import tensorflow_addons as tfa
from scipy.stats import pearsonr,spearmanr
from model_new import KerasMultiSourceGCNModel_new
import hickle as hkl
import argparse
import codecs
from subword_nmt.apply_bpe import BPE
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score

####################################Settings#################################
parser = argparse.ArgumentParser(description='Drug_response_pre')
parser.add_argument('-gpu_id', dest='gpu_id', type=str, default='7', help='GPU devices')
parser.add_argument('-use_mut', dest='use_mut', type=bool, default=True, help='use gene mutation or not')
parser.add_argument('-use_gexp', dest='use_gexp', type=bool, default=True, help='use gene expression or not')
parser.add_argument('-use_methy', dest='use_methy', type=bool, default=True, help='use methylation or not')
parser.add_argument('-use_copy', dest='use_copy', type=bool, default=True, help='use copy number or not')
parser.add_argument('-israndom', dest='israndom', type=bool, default=False, help='randomlize X and A')
# hyparameters for GCN
parser.add_argument('-unit_list', dest='unit_list', nargs='+', type=int, default=[128, 128, 128],
                    help='unit list for GCN')
parser.add_argument('-unit_edge_list', dest='unit_edge_list', nargs='+', type=int, default=[32, 32, 32],
                    help='unit list for edge GCN')
parser.add_argument('-Max_atoms', dest='Max_atoms', type=int, default=100, help='molecule padding size')
parser.add_argument('-batch_size_set', dest='batch_size_set', type=int, default=256, help='batch_size_set')
parser.add_argument('-epoch_set', dest='epoch_set', type=int, default=500, help='max epoch')
parser.add_argument('-Dropout_rate', dest='Dropout_rate', type=float, default=0.2, help='Dropout_rate')
parser.add_argument('-activation', dest='activation', type=str, default='gelu', help='activation func')
parser.add_argument('-use_bn', dest='use_bn', type=bool, default=True, help='use batchnormalization for GCN')
parser.add_argument('-use_relu', dest='use_relu', type=bool, default=True, help='use relu for GCN')
parser.add_argument('-use_GMP', dest='use_GMP', type=bool, help='use GlobalMaxPooling for GCN')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
use_mut, use_gexp, use_methy,use_copy = args.use_mut, args.use_gexp, args.use_methy,args.use_copy
israndom = args.israndom
model_suffix = ('with_mut' if use_mut else 'without_mut')+'_'+('with_gexp' if use_gexp else 'without_gexp')+'_'+(
    'with_methy' if use_methy else 'without_methy')+(
    'with_copy' if use_copy else 'without_copy')

GCN_deploy = '_'.join(map(str, args.unit_list))+'_'+('bn' if args.use_bn else 'no_bn')+'_'+(
    'relu' if args.use_relu else 'tanh')+'_'+('GMP' if args.use_GMP else 'GAP')
model_suffix = model_suffix+'_'+GCN_deploy

####################################Constants Settings###########################
TCGA_label_set = ["ALL", "BLCA", "BRCA", "CESC", "DLBC", "LIHC", "LUAD",
                  "ESCA", "GBM", "HNSC", "KIRC", "LAML", "LCML", "LGG",
                  "LUSC", "MESO", "MM", "NB", "OV", "PAAD", "SCLC", "SKCM",
                  "STAD", "THCA", 'COAD/READ']
DPATH = '../data'
Drug_info_file = '%s/GDSC/Drug_listMon Apr 10 09_29_53 2023.csv' % DPATH
Cell_line_info_file = '%s/CCLE/Cell_lines_annotations_20181226.txt' % DPATH
Drug_feature_file = '%s/GDSC/50 and 11' % DPATH
Cancer_response_exp_file = '%s/CCLE/GDSC_IC50.csv' % DPATH
Gene_info_file='%s/CCLE/Result_StandardScaler.csv' % DPATH
def MetadataGenerate(Drug_info_file, Cell_line_info_file, Drug_feature_file,
                     Gene_info, filtered):
    # drug_id --> pubchem_id
    reader = csv.reader(open(Drug_info_file, 'r'))
    rows = [item for item in reader]
    drugid2pubchemid = {item[0]: item[5] for item in rows if item[5].isdigit()}
    # map cellline --> cancer type
    cellline2cancertype = {}
    for line in open(Cell_line_info_file).readlines()[1:]:
        cellline_id = line.split('\t')[1]
        TCGA_label = line.strip().split('\t')[-1]
        # if TCGA_label in TCGA_label_set:
        cellline2cancertype[cellline_id] = TCGA_label

    # load drug features
    drug_pubchem_id_set = []
    drug_feature = {}
    for each in os.listdir(Drug_feature_file):
        drug_pubchem_id_set.append(each.split('.')[0])
        node_features_Mol, lcq_adj, edges_feature, smiles_feature = hkl.load('%s/%s' % (Drug_feature_file, each))
        # features_aug=hkl.load('%s/%s' % (Drug_feature_file, each))
        drug_feature[each.split('.')[0]] = [node_features_Mol, lcq_adj, edges_feature,smiles_feature]
        # drug_feature[each.split('.')[0]] = features_aug

    assert len(drug_pubchem_id_set) == len(drug_feature.values())
    experiment_data = pd.read_csv(Cancer_response_exp_file, sep=',', header=0, index_col=[0])
    # filter experiment data
    drug_match_list = [item for item in experiment_data.index if item.split(':')[1] in drugid2pubchemid.keys()]
    res=[item for item in experiment_data.index ]
    experiment_data_filtered = experiment_data.loc[drug_match_list]
    data_idx = []
    gene_info=pd.read_csv(Gene_info_file, sep=',', index_col=[0])
    for each_drug in experiment_data_filtered.index:
        for each_cellline in experiment_data_filtered.columns:
            pubchem_id = drugid2pubchemid[each_drug.split(':')[-1]]
            if str(pubchem_id) in drug_pubchem_id_set and each_cellline in gene_info.columns:
                if not np.isnan(experiment_data_filtered.loc[each_drug, each_cellline]) and each_cellline in cellline2cancertype.keys():
                    ln_IC50 = float(experiment_data_filtered.loc[each_drug, each_cellline])
                    data_idx.append((each_cellline, pubchem_id, ln_IC50, cellline2cancertype[each_cellline]))



    for each_drug in experiment_data_filtered.index:
        for each_cellline in experiment_data_filtered.columns:
            pubchem_id = drugid2pubchemid[each_drug.split(':')[-1]]
            for aug_data in drug_pubchem_id_set:
                if aug_data.startswith("k"):
                    pubchem_id_aug_tmp = aug_data.split(":")[0]
                    pubchem_id_aug=pubchem_id_aug_tmp.split("_")[1]
                    if aug_data.startswith("k") and pubchem_id_aug==str(pubchem_id) and str(pubchem_id) in drug_pubchem_id_set and each_cellline in gene_info.columns:
                        if not np.isnan(experiment_data_filtered.loc[each_drug, each_cellline]) and each_cellline in cellline2cancertype.keys():
                            ln_IC50 = float(experiment_data_filtered.loc[each_drug, each_cellline])
                            data_idx.append((each_cellline, aug_data, ln_IC50, cellline2cancertype[each_cellline]))


    nb_celllines = len(set([item[0] for item in data_idx]))
    nb_drugs = len(set([item[1] for item in data_idx]))
    print('%d instances across %d cell lines and %d drugs were generated.' % (len(data_idx), nb_celllines, nb_drugs))
    return  drug_feature, gene_info, data_idx

# def DataSplit(data_idx, ratio=0.95):
#     data_train_idx, data_test_idx = [], []
#     for each_type in TCGA_label_set:
#         data_subtype_idx = [item for item in data_idx if item[-1] == each_type]
#         train_list = random.sample(data_subtype_idx, int(ratio * len(data_subtype_idx)))
#         test_list = [item for item in data_subtype_idx if item not in train_list]
#         data_train_idx += train_list
#         data_test_idx += test_list
#     assert len(data_test_idx) >= args.batch_size_set
#     print(len(data_train_idx),len(data_test_idx),"?")
#     return data_train_idx, data_test_idx

def split_list(lst, n):

    split_index = int(len(lst) * (1/n))   # 计算分割点索引
    aug_list=lst[split_index:]
    result = []
    l=n-1
    for i in range(l):
        result.append(aug_list[i::l])

    return lst[:split_index],result

def DataSplit(data_idx, ratio=0.95):
    data_true_idx,aug_list=split_list(data_idx,1)
    data_train_idx, data_test_idx = [], []
    for each_type in TCGA_label_set:
        data_subtype_idx = [item for item in data_true_idx if item[-1] == each_type]
        selected_indices = random.sample(range(len(data_subtype_idx)), int(ratio * len(data_subtype_idx)))
        train_list = [data_subtype_idx[i] for i in selected_indices]
        test_list = [data_subtype_idx[i] for i in range(len(data_subtype_idx)) if i not in selected_indices]
        data_train_idx += train_list
        data_test_idx += test_list
        for data_subtype_idx_subaug in aug_list:
            data_subtype_idx_each_aug = [item for item in data_subtype_idx_subaug if item[-1] == each_type]
            train_list_aug = [data_subtype_idx_each_aug[i] for i in selected_indices]
            data_train_idx += train_list_aug
    print(len(data_train_idx),len(data_test_idx),"?")
    return data_train_idx, data_test_idx


def features_padding(node_feature, edges_feature, size):
    node_pad = np.pad(node_feature, ((0, args.Max_atoms-node_feature.shape[0]), (0, 0)), 'constant')
    edge_pad = np.pad(edges_feature,
                      ((0, args.Max_atoms-edges_feature.shape[0]), (0, args.Max_atoms-edges_feature.shape[0]), (0, 0)),
                      'constant')
    return [node_pad, edge_pad]
def _drug2emb_encoder(smile):
    # vocab_path = "{}/ESPF/drug_codes_chembl_freq_1500.txt".format(self.vocab_dir)
    # sub_csv = pd.read_csv("{}/ESPF/subword_units_map_chembl_freq_1500.csv".format(self.vocab_dir))
    vocab_path = "ESPF/drug_codes_chembl_freq_1500.txt"
    sub_csv = pd.read_csv("ESPF/subword_units_map_chembl_freq_1500.csv")
    bpe_codes_drug = codecs.open(vocab_path)
    dbpe = BPE(bpe_codes_drug, merges=-1, separator='')

    idx2word_d = sub_csv['index'].values
    words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

    max_d = 50
    t1 = dbpe.process_line(smile).split()  # split
    try:
        i1 = np.asarray([words2idx_d[i] for i in t1])  # index
    except:
        i1 = np.array([0])

    l = len(i1)
    if l < max_d:
        i = np.pad(i1, (0, max_d-l), 'constant', constant_values=0)
        input_mask = ([1] * l)+([0] * (max_d-l))
    else:
        i = i1[:max_d]
        input_mask = [1] * max_d

    return i, np.asarray(input_mask)
def FeatureExtract(data_idx, drug_feature, gene_feature):
    cancer_type_list = []
    nb_instance = len(data_idx)
    drug_data = [[] for item in range(nb_instance)]
    target = np.zeros(nb_instance, dtype='float32')
    gene_data=[]
    for idx in range(nb_instance):
        cell_line_id, pubchem_id, ln_IC50, cancer_type = data_idx[idx]
        # modify

        feat_mat, adj_list, _ ,smiles_feature= drug_feature[str(pubchem_id)]
        # fill drug data,padding to the same size with zeros
        drug_data[idx] = features_padding(feat_mat, adj_list, args.Max_atoms)
        drug_data[idx].append(smiles_feature[0])
        drug_data[idx].append(smiles_feature[1])
        # randomlize X A
        target[idx] = ln_IC50
        cell_line_info_tmp=gene_feature[cell_line_id].values
        cell_line_info_tmp_afterprocess=[]
        for geneinfo in cell_line_info_tmp:
            cell_line_info_tmp_afterprocess.append(eval(geneinfo))
        gene_data.append(cell_line_info_tmp_afterprocess)
        cancer_type_list.append([cancer_type, cell_line_id, pubchem_id])
    gene_data=np.array(gene_data)
    return drug_data, gene_data, target, cancer_type_list

class MyCallback(Callback):
    def __init__(self, validation_data, patience):
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        self.best_weight = None
        self.patience = patience
    def on_train_begin(self, logs={}):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = -np.Inf
        print("begin")
        return
    def on_train_end(self, logs={}):
        self.model.set_weights(self.best_weight)
        self.model.save('MyBestDeepAEG_%s.h5' % self.best)

        if self.stopped_epoch > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch+1))
        return
    def on_epoch_begin(self, epoch, logs={}):
        return
    def on_epoch_end(self, epoch, logs={}):
        y_pred_val = self.model.predict(self.x_val, batch_size=args.batch_size_set)
        pcc_val = pearsonr(self.y_val, y_pred_val[:, 0])[0]
        print('pcc-val: %s' % str(round(pcc_val, 4)))
        if pcc_val > self.best:
            self.best = pcc_val
            self.wait = 0
            self.best_weight = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
        return

def generate_batch_data(data_idx, batch_size, drug_feature, gene_feature):
    np.random.shuffle(data_idx)
    while True:
        if len(data_idx) % batch_size == 0:
            times_valid = len(data_idx) // batch_size
        else:
            times_valid = len(data_idx) // batch_size+1
        for step_valid in range(times_valid):
            if (step_valid+1) * batch_size > len(data_idx):
                present_idx = data_idx[len(data_idx)-batch_size:]
            else:
                present_idx = data_idx[step_valid * batch_size:(step_valid+1) * batch_size]
            X_drug_data_train, X_cell_line_test, Y_train, cancer_type_train_list \
                = FeatureExtract(present_idx, drug_feature, gene_feature)
            X_drug_feat_data_train = [item[0] for item in X_drug_data_train]
            X_drug_adj_data_train = [item[1] for item in X_drug_data_train]
            X_drug_smiles_data_train = [item[2] for item in X_drug_data_train]
            X_drug_smiles_mask_data_train = [item[3] for item in X_drug_data_train]

            X_drug_feat_data_train = np.array(X_drug_feat_data_train)  # nb_instance * Max_stom * feat_dimeag
            X_drug_adj_data_train = np.array(X_drug_adj_data_train)  # nb_instance * Max_stom * Max_stom
            X_drug_smiles_data_train = np.array(X_drug_smiles_data_train)
            X_drug_smiles_mask_data_train = np.array(X_drug_smiles_mask_data_train)

            copy=X_cell_line_test[...,0]
            gexpr=X_cell_line_test[...,1]
            mutation=X_cell_line_test[...,2]
            methy=X_cell_line_test[...,3]
            yield [X_drug_feat_data_train, X_drug_adj_data_train,X_drug_smiles_data_train,X_drug_smiles_mask_data_train, copy,mutation,gexpr,methy], Y_train
                # yield [X_drug_feat_data_train, X_drug_adj_data_train, X_cell_line_test], Y_train



def ModelTraining(model, data_train_idx, drug_feature, gene_feature,
                  data_test_idx, validation_data, nb_epoch=args.epoch_set, batch_size=args.batch_size_set):
    # optimizer=tfa.optimizers.AdamW(learning_rate=0.001,beta_1=0.9, beta_2=0.999,epsilon=None,weight_decay=0.0,amsgrad=False)
    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'],run_eagerly=True)
    callbacks = [ModelCheckpoint('best_DeepAEG_%s.h5' % model_suffix, monitor='val_loss', save_best_only=False,
                                 save_weights_only=False),
                 MyCallback(validation_data=validation_data, patience=20)]
    if len(data_train_idx) % args.batch_size_set == 0:
        steps_per_epoch = len(data_train_idx) // args.batch_size_set
    else:
        steps_per_epoch = len(data_train_idx) // args.batch_size_set+1
    if len(data_test_idx) % args.batch_size_set == 0:
        validation_steps = len(data_test_idx) // args.batch_size_set
    else:
        validation_steps = len(data_test_idx) // args.batch_size_set+1
    training_generator = generate_batch_data(data_train_idx, batch_size=args.batch_size_set, drug_feature=drug_feature,gene_feature=gene_feature)

    model.fit(x=training_generator, steps_per_epoch=steps_per_epoch, epochs=nb_epoch, verbose=1,callbacks=callbacks, validation_data=validation_data, validation_steps=validation_steps,use_multiprocessing=False)
    # model.fit(x=[X_drug_feat_data_train,X_drug_adj_data_train,X_mutation_data_train,X_gexpr_data_train,X_methylation_data_train],y=Y_train,batch_size=1,epochs=nb_epoch,validation_split=0,callbacks=callbacks)
    return model

def ModelEvaluate(model, X_drug_data_test,  X_cell_line_test, Y_test):
    X_drug_feat_data_test = [item[0] for item in X_drug_data_test]
    X_drug_adj_data_test = [item[1] for item in X_drug_data_test]
    X_drug_smiles_data_test = [item[2] for item in X_drug_data_test]
    X_drug_smiles_mask_data_test = [item[3] for item in X_drug_data_test]

    X_drug_feat_data_test = np.array(X_drug_feat_data_test)  # nb_instance * Max_stom * feat_dim
    X_drug_adj_data_test = np.array(X_drug_adj_data_test)  # nb_instance * Max_stom * Max_stom
    X_drug_smiles_data_test=np.array(X_drug_smiles_data_test)
    X_drug_smiles_mask_data_test=np.array(X_drug_smiles_mask_data_test)

    copy = X_cell_line_test[..., 0]
    gexpr = X_cell_line_test[..., 1]
    mutation = X_cell_line_test[..., 2]
    methy = X_cell_line_test[..., 3]

    Y_pred = model.predict([X_drug_feat_data_test, X_drug_adj_data_test,X_drug_smiles_data_test,X_drug_smiles_mask_data_test,
                            copy,mutation,gexpr,methy],batch_size=args.batch_size_set)
    import csv
    file_path = "2.csv"
    if not os.path.isfile(file_path):
        with open('2.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            for i in range(len(Y_test)):
                writer.writerow([Y_pred[:, 0][i],Y_test[i]])
    overall_pcc = pearsonr(Y_pred[:, 0], Y_test)[0]

    from math import sqrt
    mse = np.sum((Y_test - Y_pred[:, 0]) ** 2) / len(Y_test)
    rmse = sqrt(mse)
    spearmanr_pcc=spearmanr(Y_pred[:, 0], Y_test)[0]

    print("The overall Pearson's correlation is %.4f." % overall_pcc)
    print("The overall rmse's  is %.4f." % rmse)
    print("The overall spearmanr_pcc's  is %.4f." % spearmanr_pcc)


    # with open('result0.1tf2.log', 'a') as f:
    #     f.write(str(overall_pcc)+" overall_pcc"+"0.1 tf2\n")

# 
def main():
    random.seed(0)
    #all features with aug
    drug_feature, gene_feature, data_idx = MetadataGenerate\
        (Drug_info_file,Cell_line_info_file,Drug_feature_file,Gene_info_file,False)

    data_train_idx, data_test_idx = DataSplit(data_idx)
    X_drug_data_test, X_cell_line_test, Y_test, cancer_type_test_list=\
        FeatureExtract(data_test_idx,drug_feature,gene_feature)
    X_drug_feat_data_test = [item[0] for item in X_drug_data_test]
    X_drug_adj_data_test = [item[1] for item in X_drug_data_test]
    X_drug_smiles_data_test = [item[2] for item in X_drug_data_test]
    X_drug_smiles_mask_data_test = [item[3] for item in X_drug_data_test]


    X_drug_feat_data_test = np.array(X_drug_feat_data_test)  # nb_instance * Max_stom * feat_dim
    X_drug_adj_data_test = np.array(X_drug_adj_data_test)
    X_drug_smiles_data_test=np.array(X_drug_smiles_data_test)
    X_drug_smiles_mask_data_test=np.array(X_drug_smiles_mask_data_test)

    copy = X_cell_line_test[..., 0]
    gexpr = X_cell_line_test[..., 1]
    mutation = X_cell_line_test[..., 2]
    methy = X_cell_line_test[..., 3]
    validation_data = ([X_drug_feat_data_test, X_drug_adj_data_test,X_drug_smiles_data_test,X_drug_smiles_mask_data_test,copy,mutation,gexpr,methy],Y_test)
    print(Y_test[-1])

    model = KerasMultiSourceGCNModel_new(use_mut, use_gexp, use_methy,use_copy).createMaster(X_drug_data_test[0][0].shape[-1],
                                                                                X_drug_data_test[0][1].shape[-1],
                                                                                X_drug_data_test[0][2].shape[-1],
                                                                                X_drug_data_test[0][3].shape[-1],
                                                                                X_cell_line_test.shape,
                                                                                args.unit_list, args.unit_edge_list,
                                                                                args.batch_size_set, args.Dropout_rate,
                                                                                args.activation, args.use_relu,
                                                                                args.use_bn, args.use_GMP)
    print('Begin training...')
    model = ModelTraining(model, data_train_idx, drug_feature, gene_feature,data_test_idx, validation_data, nb_epoch=args.epoch_set, batch_size=args.batch_size_set)

    # model = loadmodel()
    # optimizer = tfa.optimizers.AdamW(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, weight_decay=0.0,
    #                                  amsgrad=False)
    # model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'], run_eagerly=True)
    ModelEvaluate(model, X_drug_data_test, X_cell_line_test, Y_test)
if __name__ == '__main__':
    main()