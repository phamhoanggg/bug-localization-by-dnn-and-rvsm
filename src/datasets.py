from collections import namedtuple
from pathlib import Path

# Dataset root directory
_DATASET_ROOT = Path('../data')

Dataset = namedtuple('Dataset',
[
    'name',
    'src',
    'bug_repo_txt',     #bug report text
    'bug_repo_csv',     #bug report csv
    'src_pickle',     #source code pickle
    'features'
])

# Source codes and bug repositories

aspectj = Dataset(
    'aspectj',
    _DATASET_ROOT / 'org.aspectj-bug43351/',
    _DATASET_ROOT / 'AspectJ.txt',
    './aspectj_bug_reports.csv',
    _DATASET_ROOT / 'aspectj_src.pickle',
    _DATASET_ROOT / 'features_aspectj.csv'
)

eclipse = Dataset(
    'eclipse',
    _DATASET_ROOT / 'eclipse.platform.ui-johna-402445/',
    _DATASET_ROOT / 'Eclipse_Platform_UI.txt',
    './eclipse_bug_reports.csv',
    _DATASET_ROOT / 'eclipse_src.pickle',
    _DATASET_ROOT / 'features_eclipse_base.csv'
)

swt = Dataset(
    'swt',
    _DATASET_ROOT / 'eclipse.platform.swt-xulrunner-31/',
    _DATASET_ROOT / 'SWT.txt',
    './swt_bug_reports.csv',
    _DATASET_ROOT / 'swt_src.pickle',
    _DATASET_ROOT / 'features_swt.csv'
)

tomcat = Dataset(
    'tomcat',
    _DATASET_ROOT / 'tomcat-7.0.51/',
    _DATASET_ROOT / 'Tomcat.txt',
    './tomcat_bug_reports.csv',
    _DATASET_ROOT / 'tomcat_src.pickle',
    _DATASET_ROOT / 'features_tomcat.csv/'
)


### Current dataset in use. (change this name to change the dataset)
DATASET = tomcat
# if __name__ == '__main__':
#     print(DATASETs.name, DATASETs.src, DATASET.bug_repo)
