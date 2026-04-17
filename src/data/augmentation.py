import pandas as pd
import ast
import snappy

SIGN_FLIP_TASKS = {'Signature', 'Rasmussen s', 'Ozsvath-Szabo tau'}
MIRROR_SAFE_TASKS = {'Alternating', 'Crossing Number', 'Unknotting Number', 'Determinant',
    'Genus-3D', 'Genus-4D', 'Genus-4D (Top.)', 'Arf' }

def fix_pd_notation(pd_notation: str) -> str:
    return pd_notation.replace(';', ',')


def _mirror_target_value(value, task_name: str):
    if task_name in MIRROR_SAFE_TASKS:
        return value
    if task_name in SIGN_FLIP_TASKS:
        return -int(float(value))
    raise ValueError(f'Unknown task for mirror behavior: {task_name}')


def augment_knot_split(split_df: pd.DataFrame, task_name: str, target_col: str, many_target: int = 10, many_tries: int = 100, 
    many_method: str = 'backtrack', include_mirror: bool = True) -> pd.DataFrame:

    if 'PD Notation' not in split_df.columns:
        raise ValueError("Input DataFrame must contain 'PD Notation' column")
    if target_col not in split_df.columns:
        raise ValueError(f"Input DataFrame must contain target column '{target_col}'")
    if many_target < 1:
        raise ValueError('many_target must be >= 1')
    if many_tries < 1:
        raise ValueError('many_tries must be >= 1')
    if many_method not in {'backtrack', 'exterior'}:
        raise ValueError("many_method must be 'backtrack' or 'exterior'")

    augmented_rows = []
    seen = set()
    skipped = 0

    for idx in range(len(split_df)):
        row = split_df.iloc[idx]
        pd_notation = row['PD Notation']

        if pd.isna(pd_notation):
            skipped += 1
            continue

        try:
            pd_str = fix_pd_notation(str(pd_notation))
            pd_list = ast.literal_eval(pd_str)
            base_link = snappy.Link(pd_list)

            diagrams = base_link.many_diagrams(target=many_target, tries=many_tries, method=many_method)

            for diagram in diagrams:
                aug_row = row.copy()
                aug_row['PD Notation'] = str(diagram.PD_code())

                key = (aug_row['PD Notation'], aug_row[target_col])
                if key not in seen:
                    seen.add(key)
                    augmented_rows.append(aug_row)

                if include_mirror:
                    mirror = diagram.mirror()
                    mirror_row = row.copy()
                    mirror_row['PD Notation'] = str(mirror.PD_code())
                    mirror_row[target_col] = _mirror_target_value(row[target_col], task_name)

                    mirror_key = (mirror_row['PD Notation'], mirror_row[target_col])
                    if mirror_key not in seen:
                        seen.add(mirror_key)
                        augmented_rows.append(mirror_row)

        except Exception as e:
            skipped += 1
            print(f'Row {idx} failed: {e}')

    aug_df = pd.DataFrame(augmented_rows).reset_index(drop=True)
    print(f'Input rows: {len(split_df)}')
    print(f'Augmented rows: {len(aug_df)}')
    print(f'Skipped source rows: {skipped}')
    
    return aug_df