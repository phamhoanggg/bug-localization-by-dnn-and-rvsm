import pickle
import csv
from preprocessing import BugReport, SourceFile

# Load the pickle file
pickleFile = '../data/preprocessed_src.pickle'
with open(pickleFile, 'rb') as f:
    src_files = pickle.load(f)

# Prepare data for CSV
rows = []

if isinstance(src_files, dict):
    # It's a dictionary of SrcFile
    for key, src in src_files.items():
        if isinstance(src, SourceFile):
            rows.append({
                'all_content': src.all_content,
                'comments': src.comments,
                'class_names': src.class_names,
                'attributes': src.attributes,
                'method_names': src.method_names,
                'variables': src.variables,
                'file_name': src.file_name,
                'pos_tagged_comments': src.pos_tagged_comments,
                'exact_file_name': src.exact_file_name,
                'package_name': src.package_name
            })
elif isinstance(src_files, list):
    # It's a list of SrcFile
    for src in src_files:
        if isinstance(src, SourceFile):
            rows.append({
                'all_content': src.all_content,
                'comments': src.comments,
                'class_names': src.class_names,
                'attributes': src.attributes,
                'method_names': src.method_names,
                'variables': src.variables,
                'file_name': src.file_name,
                'pos_tagged_comments': src.pos_tagged_comments,
                'exact_file_name': src.exact_file_name,
                'package_name': src.package_name
            })
elif isinstance(src_files, SourceFile):
    # Single SrcFile object
    rows.append({
        'all_content': src_files.all_content,
        'comments': src_files.comments,
        'class_names': src_files.class_names,
        'attributes': src_files.attributes,
        'method_names': src_files.method_names,
        'variables': src_files.variables,
        'file_name': src_files.file_name,
        'pos_tagged_comments': src_files.pos_tagged_comments,
        'exact_file_name': src_files.exact_file_name,
        'package_name': src_files.package_name
    })
else:
    raise TypeError("Unsupported type for src_files!")

# Write to CSV
csv_file = 'src_file.csv'
fieldnames = ['all_content', 'comments', 'class_names', 'attributes', 'method_names', 'variables', 'file_name',
                 'pos_tagged_comments', 'exact_file_name', 'package_name']

with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved {len(rows)} src files to {csv_file}!")