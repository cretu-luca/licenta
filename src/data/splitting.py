from sklearn.model_selection import GroupShuffleSplit
import numpy as np

def five_splits_by_knot_name(knot_names, proportions=(0.5, 0.25, 0.25), seeds=(0,1,2,3,4)):
    p_train, p_val, p_test = proportions
    groups = np.asarray(knot_names); idx = np.arange(len(groups))
    out = []
    for seed in seeds:
        train, rest = next(GroupShuffleSplit(1, train_size=p_train, random_state=seed)
                           .split(idx, groups=groups))
        rel_val = p_val / (p_val + p_test)
        v_loc, t_loc = next(GroupShuffleSplit(1, train_size=rel_val, random_state=seed)
                            .split(rest, groups=groups[rest]))
        out.append((train, rest[v_loc], rest[t_loc]))
    return out