from utils.read_data import load_similarity_mat
from numpy import triu, ones


def pd_get_upper(df):
    keep = triu(ones(df.shape), 1).astype('bool').reshape(df.size)
    return df.stack()[keep]

def pd_fix_columns(x):
    x.columns = x.columns.str.split('.').str[0]
    return x

def pd_fix_words(n_syl, x):
    sim_path = f'results/f1_similarity_mat_normalize_false_nsyl{str(n_syl)}.csv'
    ref = load_similarity_mat(sim_path)
    x.index = ref.index
    return pd_fix_columns(x)