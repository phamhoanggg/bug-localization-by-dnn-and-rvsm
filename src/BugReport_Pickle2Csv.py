import pickle
import csv
from preprocessing import BugReport, SourceFile

# Load the pickle file
pickleFile = '../data/preprocessed_reports.pickle'
with open(pickleFile, 'rb') as f:
    bug_reports = pickle.load(f)

# Prepare data for CSV
rows = []

if isinstance(bug_reports, dict):
    # It's a dictionary of BugReports
    for key, report in bug_reports.items():
        if isinstance(report, BugReport):
            rows.append({
                'summary': report.summary,
                'description': report.description,
                'fixed_files': '; '.join(report.fixed_files) if report.fixed_files else '',
                'report_time': report.report_time,
                'pos_tagged_summary': report.pos_tagged_summary,
                'pos_tagged_description': report.pos_tagged_description,
                'stack_traces': report.stack_traces,
                'stack_traces_remove': report.stack_traces_remove
            })
elif isinstance(bug_reports, list):
    # It's a list of BugReports
    for report in bug_reports:
        if isinstance(report, BugReport):
            rows.append({
                'summary': report.summary,
                'description': report.description,
                'fixed_files': '; '.join(report.fixed_files) if report.fixed_files else '',
                'report_time': report.report_time,
                'pos_tagged_summary': report.pos_tagged_summary,
                'pos_tagged_description': report.pos_tagged_description,
                'stack_traces': report.stack_traces,
                'stack_traces_remove': report.stack_traces_remove
            })
elif isinstance(bug_reports, BugReport):
    # Single BugReport object
    rows.append({
        'summary': bug_reports.summary,
        'description': bug_reports.description,
        'fixed_files': '; '.join(bug_reports.fixed_files) if bug_reports.fixed_files else '',
        'report_time': bug_reports.report_time,
        'pos_tagged_summary': bug_reports.pos_tagged_summary,
        'pos_tagged_description': bug_reports.pos_tagged_description,
        'stack_traces': bug_reports.stack_traces,
        'stack_traces_remove': bug_reports.stack_traces_remove
    })
else:
    raise TypeError("Unsupported type for bug_reports!")

# Write to CSV
csv_file = 'bug_reports.csv'
fieldnames = ['summary', 'description', 'fixed_files', 'report_time', 'pos_tagged_summary', 'pos_tagged_description','stack_traces','stack_traces_remove']


with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved {len(rows)} reports to {csv_file}!")