"""Execute A1 offline, check relocation, export HTML and save audit metrics."""

from datetime import datetime, timezone
from pathlib import Path
import base64
import json
import os
import shutil
import tempfile

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'base' / 'EDA_food_inspections.ipynb'
SNAPSHOT = 'food_inspections_2016-01-01_2026-10-25.csv'

OFFLINE_GUARD = """
import socket
_original_connect = socket.socket.connect
def _offline_connect(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        if str(address[0]) not in ('127.0.0.1', '::1', 'localhost'):
            raise RuntimeError('External network disabled during verification: ' + str(address[0]))
    return _original_connect(self, address)
socket.socket.connect = _offline_connect
"""

METRICS = """
import json
revision_metrics = {
    'raw_rows': len(raw), 'labelled_rows': len(labelled),
    'missing_licences': int(missing_licence.sum()), 'zero_licences': int(zero_licence.sum()),
    'other_invalid_licences': int(other_invalid.sum()),
    'zero_names': int(raw.loc[zero_licence, 'DBA Name'].nunique()),
    'zero_addresses': int(raw.loc[zero_licence, 'Address'].nunique()),
    'max_valid_licence_visits': int(positive_visits.max()),
    'same_day_groups_valid': same_day_groups, 'same_day_rows_valid': same_day_rows,
    'training_labelled_rows': len(eda), 'training_fail_rate': float(training_fail_rate),
    'history_comparison_omitted_unknown': history_omitted,
    'risk_standardisation': standardisation.reset_index().to_dict(orient='records'),
    'history_by_prior': prior_stats.reset_index().astype({'prior_group':'str'}).to_dict(orient='records'),
    'history_by_gap': gap_stats.reset_index().astype({'gap_group':'str'}).to_dict(orient='records'),
    'split_summary': split_summary.to_dict(orient='records'),
}
print(json.dumps(revision_metrics, ensure_ascii=False))
"""


def execute(cwd, notebook_path=NOTEBOOK):
    notebook = nbformat.read(notebook_path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    notebook.cells.insert(0, nbformat.v4.new_code_cell(OFFLINE_GUARD))
    notebook.cells.append(nbformat.v4.new_code_cell(METRICS))
    NotebookClient(notebook, timeout=240, kernel_name='python3',
                   resources={'metadata': {'path': str(cwd)}}).execute()
    metrics_text = ''.join(o.text for o in notebook.cells[-1].outputs if o.output_type == 'stream')
    metrics = json.loads(metrics_text)
    notebook.cells = notebook.cells[1:-1]
    for count, cell in enumerate((c for c in notebook.cells if c.cell_type == 'code'), start=1):
        cell.execution_count = count
    errors = [o for c in notebook.cells for o in c.get('outputs', []) if o.output_type == 'error']
    figures = [o.data['image/png'] for c in notebook.cells for o in c.get('outputs', [])
               if 'image/png' in o.get('data', {})]
    assert not errors
    assert len(figures) == 8
    nbformat.validate(notebook)
    return notebook, metrics, figures


executed, metrics, figures = execute(ROOT / 'base')
print('PASS: empty-kernel execution from base/, external network blocked.', flush=True)

# Hard links avoid copying a large immutable CSV; notebook paths resolve only in the moved tree.
with tempfile.TemporaryDirectory(prefix='dsaa1001_relocation_') as temporary:
    moved = Path(temporary)
    (moved / 'base').mkdir()
    (moved / 'data').mkdir()
    shutil.copy2(NOTEBOOK, moved / 'base' / NOTEBOOK.name)
    stored_snapshot = SNAPSHOT if (ROOT / 'data' / SNAPSHOT).is_file() else SNAPSHOT + '.gz'
    for name in [stored_snapshot, 'download_metadata.json']:
        os.link(ROOT / 'data' / name, moved / 'data' / name)
    moved_notebook, moved_metrics, moved_figures = execute(moved, moved / 'base' / NOTEBOOK.name)
    assert metrics == moved_metrics
    assert figures == moved_figures
print('PASS: relocated folder, project-root execution, identical metrics and figures.', flush=True)

nbformat.write(executed, NOTEBOOK)
(ROOT / 'base' / 'eda_revision_metrics.json').write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

exporter = HTMLExporter(template_name='lab')
exporter.embed_images = True
html, _ = exporter.from_notebook_node(executed)
soup = BeautifulSoup(html, 'html.parser')
for tag in soup.find_all('script', src=True):
    if tag['src'].startswith(('http://', 'https://', '//')):
        tag.decompose()
for tag in soup.find_all('link', href=True):
    if tag['href'].startswith(('http://', 'https://', '//')):
        tag.decompose()
assert len(soup.select('img[src^="data:image/png"]')) == 8
(ROOT / 'base' / 'EDA_food_inspections_revised.html').write_text(str(soup), encoding='utf-8')

figure_folder = Path(tempfile.gettempdir()) / 'dsaa1001_revised_figures'
figure_folder.mkdir(exist_ok=True)
for number, image in enumerate(figures, start=1):
    (figure_folder / f'figure_{number:02d}.png').write_bytes(base64.b64decode(image))

verification = {
    'verified_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    'notebook': 'base/EDA_food_inspections.ipynb',
    'cells': len(executed.cells),
    'executed_code_cells': sum(c.cell_type == 'code' for c in executed.cells),
    'figure_outputs': len(figures),
    'external_network_blocked': True,
    'base_directory_run': 'passed',
    'relocated_project_root_run': 'passed',
    'metrics_and_figures_identical': True,
    'relocation_input_method': 'Hard-linked immutable CSV and metadata in a temporary project tree.',
    'environment_scope': 'Fresh kernels in the installed Python environment; not a newly provisioned dependency environment.',
}
(ROOT / 'base' / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n', encoding='utf-8')
print(json.dumps(verification, indent=2))
print('Figure files for visual review:', figure_folder)
