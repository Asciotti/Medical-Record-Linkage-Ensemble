import recordlinkage as rl, pandas as pd, numpy as np
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from recordlinkage.preprocessing import phonetic
from numpy.random import choice
import collections, numpy
from IPython.display import clear_output
from sklearn.model_selection import train_test_split, KFold

# (andrew) Need to add above directory to path since these notebooks run in their relative
# directory
import sys
sys.path.append("..")
from utils import (
    generate_true_links,
    generate_false_links,
    swap_fields_flag,
    join_names_space,
    join_names_dash,
    abb_surname,
    reset_day,
    set_random_seed
)
from training_utils import train_model, classify, evaluation, blocking_performance

set_random_seed()

trainset = 'febrl_UNSW_train'
testset = 'febrl_UNSW_test'

## I did not touch these yet b/c there are differences
## - Andrew
def extract_features(df, links):
    c = rl.Compare()
    c.string('given_name', 'given_name', method='jarowinkler', label='y_name')
    c.string('given_name_soundex', 'given_name_soundex', method='jarowinkler', label='y_name_soundex')
    c.string('given_name_nysiis', 'given_name_nysiis', method='jarowinkler', label='y_name_nysiis')
    c.string('surname', 'surname', method='jarowinkler', label='y_surname')
    c.string('surname_soundex', 'surname_soundex', method='jarowinkler', label='y_surname_soundex')
    c.string('surname_nysiis', 'surname_nysiis', method='jarowinkler', label='y_surname_nysiis')
    c.exact('street_number', 'street_number', label='y_street_number')
    c.string('address_1', 'address_1', method='levenshtein', threshold=0.7, label='y_address1')
    c.string('address_2', 'address_2', method='levenshtein', threshold=0.7, label='y_address2')
    c.exact('postcode', 'postcode', label='y_postcode')
    c.exact('day', 'day', label='y_day')
    c.exact('month', 'month', label='y_month')
    c.exact('year', 'year', label='y_year')
        
    # Build features
    feature_vectors = c.compute(links, df, df)
    return feature_vectors


def generate_train_X_y(df,train_true_links):
    # This routine is to generate the feature vector X and the corresponding labels y
    # with exactly equal number of samples for both classes to train the classifier.
    pos = extract_features(df, train_true_links)
    train_false_links = generate_false_links(df, len(train_true_links))    
    neg = extract_features(df, train_false_links)
    X = pos.values.tolist() + neg.values.tolist()
    y = [1]*len(pos)+[0]*len(neg)
    X, y = shuffle(X, y, random_state=0)
    X = np.array(X)
    y = np.array(y)
    return X, y

def prep_data():
    ## TRAIN SET CONSTRUCTION

    # Import
    print("Import train set...")
    df_train = pd.read_csv(trainset+".csv", index_col = "rec_id")
    train_true_links = generate_true_links(df_train)
    print("Train set size:", len(df_train), ", number of matched pairs: ", str(len(train_true_links)))

    # Preprocess train set
    df_train['postcode'] = df_train['postcode'].astype(str)
    df_train['given_name_soundex'] = phonetic(df_train['given_name'], method='soundex')
    df_train['given_name_nysiis'] = phonetic(df_train['given_name'], method='nysiis')
    df_train['surname_soundex'] = phonetic(df_train['surname'], method='soundex')
    df_train['surname_nysiis'] = phonetic(df_train['surname'], method='nysiis')

    # Final train feature vectors and labels
    X_train, y_train = generate_train_X_y(df_train, train_true_links)
    print("Finished building X_train, y_train")
    # Blocking Criteria: declare non-match of all of the below fields disagree
    # Import
    print("Import test set...")
    df_test = pd.read_csv(testset+".csv", index_col = "rec_id")
    test_true_links = generate_true_links(df_test)
    leng_test_true_links = len(test_true_links)
    print("Test set size:", len(df_test), ", number of matched pairs: ", str(leng_test_true_links))

    print("BLOCKING PERFORMANCE:")
    blocking_fields = ["given_name", "surname", "postcode"]
    all_candidate_pairs = []
    for field in blocking_fields:
        block_indexer = rl.BlockIndex(on=field)
        candidates = block_indexer.index(df_test)
        # Comment(alecmori): blocking_performance takes two arguments, I think it's these two
        # detects = blocking_performance(candidates, test_true_links, df_test)
        detects = blocking_performance(candidates, df_test)
        all_candidate_pairs = candidates.union(all_candidate_pairs)
        print("Number of pairs of matched "+ field +": "+str(len(candidates)), ", detected ",
            detects,'/'+ str(leng_test_true_links) + " true matched pairs, missed " + 
            str(leng_test_true_links-detects) )
    # Comment(alecmori): blocking_performance takes two arguments, I think it's these two
    # detects = blocking_performance(all_candidate_pairs, test_true_links, df_test)
    detects = blocking_performance(all_candidate_pairs, df_test)
    print("Number of pairs of at least 1 field matched: " + str(len(all_candidate_pairs)), ", detected ",
        detects,'/'+ str(leng_test_true_links) + " true matched pairs, missed " + 
            str(leng_test_true_links-detects) )


    ## TEST SET CONSTRUCTION

    # Preprocess test set
    print("Processing test set...")
    print("Preprocess...")
    df_test['postcode'] = df_test['postcode'].astype(str)
    df_test['given_name_soundex'] = phonetic(df_test['given_name'], method='soundex')
    df_test['given_name_nysiis'] = phonetic(df_test['given_name'], method='nysiis')
    df_test['surname_soundex'] = phonetic(df_test['surname'], method='soundex')
    df_test['surname_nysiis'] = phonetic(df_test['surname'], method='nysiis')

    # Test feature vectors and labels construction
    print("Extract feature vectors...")
    df_X_test = extract_features(df_test, all_candidate_pairs)
    vectors = df_X_test.values.tolist()
    labels = [0]*len(vectors)
    feature_index = df_X_test.index
    for i in range(0, len(feature_index)):
        if df_test.loc[feature_index[i][0]]["match_id"]==df_test.loc[feature_index[i][1]]["match_id"]:
            labels[i] = 1
    X_test, y_test = shuffle(vectors, labels, random_state=0)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    print("Count labels of y_test:",collections.Counter(y_test))
    print("Finished building X_test, y_test")

    return X_train, y_train, X_test, y_test

def train_and_validate(X_train,y_train, X_test, y_test):
    ## ENSEMBLE CLASSIFICATION AND EVALUATION

    print("BAGGING PERFORMANCE:\n")
    modeltypes = ['svm', 'nn', 'lg'] 
    modeltypes_2 = ['linear', 'relu', 'l2']
    modelparams = [0.005, 100, 0.2]
    nFold = 10
    kf = KFold(n_splits=nFold)
    model_raw_score = [0]*3
    model_binary_score = [0]*3
    model_i = 0
    for model_i in range(3):
        modeltype = modeltypes[model_i]
        modeltype_2 = modeltypes_2[model_i]
        modelparam = modelparams[model_i]
        print(modeltype, "per fold:")
        iFold = 0
        result_fold = [0]*nFold
        final_eval_fold = [0]*nFold
        for train_index, valid_index in kf.split(X_train):
            X_train_fold = X_train[train_index]
            y_train_fold = y_train[train_index]
            md =  train_model(modeltype, modelparam, X_train_fold, y_train_fold, modeltype_2)
            result_fold[iFold] = classify(md, X_test)
            final_eval_fold[iFold] = evaluation(y_test, result_fold[iFold])
            print("Fold", str(iFold), final_eval_fold[iFold])
            iFold = iFold + 1
        bagging_raw_score = np.average(result_fold, axis=0)
        bagging_binary_score  = np.copy(bagging_raw_score)
        bagging_binary_score[bagging_binary_score > 0.5] = 1
        bagging_binary_score[bagging_binary_score <= 0.5] = 0
        bagging_eval = evaluation(y_test, bagging_binary_score)
        print(modeltype, "bagging:", bagging_eval)
        print('')
        model_raw_score[model_i] = bagging_raw_score
        model_binary_score[model_i] = bagging_binary_score

def main():
    X_train, y_train, X_test, y_test = prep_data()
    train_and_validate(X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    from memory_profiler import memory_usage
    from multiprocessing import freeze_support  
    freeze_support()

    mem_usage = memory_usage(main)
    print(f'RAM: MAX {max(mem_usage)} AVG {np.mean(mem_usage)} MIN {min(mem_usage)}')

