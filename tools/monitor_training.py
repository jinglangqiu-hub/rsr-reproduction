"""Read the frozen experiment manifest and training logs on localhost."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
from urllib.parse import parse_qs, urlsplit


PROJECT = Path(__file__).resolve().parents[1]
NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
EPOCH = re.compile(rf'epoch:\s*(\d+)\s+time:\s*({NUMBER})\s*')
METRIC = re.compile(rf"['\"](mse|mrrt|btl)['\"]\s*:\s*(?:np\.float(?:32|64)\()?({NUMBER})(?=\s*[,}})])")


def finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError, OverflowError):
        return None


def read_json(path):
    try:
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def read_epochs(path):
    try:
        content = path.read_bytes().decode('utf-8', errors='replace')
    except OSError:
        return [], []
    lines = content[:content.rfind('\n') + 1].splitlines()
    rows, pending = [], {}
    loss_labels = {'Train Loss': 'train', 'Valid MSE': 'validation', 'Test MSE': 'test'}
    metric_labels = {'Valid preformance': 'validation', 'Valid performance': 'validation',
                     'Test performance': 'test'}
    for line in lines:
        label, _, body = line.strip().partition(':')
        if label == 'Train Loss':
            pending = {}
        if label in loss_labels:
            values = body.split()
            if len(values) == 3:
                pending.setdefault(loss_labels[label], {}).update(
                    zip(('loss', 'regression_loss', 'ranking_loss'), map(finite, values)))
        elif label in metric_labels:
            fields = {key: finite(value) for key, value in METRIC.findall(body)}
            btl = fields.get('btl')
            pending.setdefault(metric_labels[label], {}).update(
                mse=fields.get('mse'), mrr=fields.get('mrrt'),
                irr=btl - 1.0 if btl is not None else None)
        elif match := EPOCH.fullmatch(line.strip()):
            epoch = int(match[1]) + 1
            if pending and (not rows or epoch > rows[-1]['epoch']):
                rows.append({'epoch': epoch, 'seconds': finite(match[2]), **pending})
            pending = {}
    return rows, lines[-24:]


def modified(path):
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def snapshot(plan, root, requested, search):
    runs = {run['id']: run for run in plan['runs']}
    updated = {}
    completed = set()
    configurations = {}
    for run_id, run in runs.items():
        configurations[run_id] = plan['configurations'][run['configuration']]
        folder = root / run_id
        if folder.resolve().parent == root:
            timestamp = modified(folder / 'train.log') or modified(folder / 'config.json')
            if timestamp is not None:
                updated[run_id] = timestamp
                if modified(folder / 'result.json') is not None:
                    completed.add(run_id)
                saved = read_json(folder / 'config.json')
                if saved.get('run', {}).get('id') == run_id and isinstance(saved.get('configuration'), dict):
                    configurations[run_id] = {**configurations[run_id], **saved['configuration']}
    paused = search.get('active') if search.get('status') == 'paused' else None
    active = search.get('active') if search.get('status') == 'running' else None
    selected = requested if requested in runs else active or paused
    if selected not in runs:
        selected = max(updated, key=updated.get) if updated else next(iter(runs))
    run = runs[selected]
    config = configurations[selected]
    folder = root / selected
    rows, tail = read_epochs(folder / 'train.log') if selected in updated else ([], [])
    result = read_json(folder / 'result.json') if selected in updated else {}
    epochs = config['epochs']
    if selected == paused:
        status = '训练已暂停'
    elif result:
        status = '已完成'
    elif rows and rows[-1]['epoch'] >= epochs:
        status = '全部轮次已记录，等待结果文件'
    elif rows:
        status = '已记录逐轮指标'
    elif selected in updated:
        status = '已创建运行，等待首轮指标'
    else:
        status = '等待训练开始'
    valid_rows = [row for row in rows if row.get('validation', {}).get('loss') is not None]
    best = min(valid_rows, key=lambda row: row['validation']['loss'], default=None)
    menu_fields = ('market', 'relation', 'inner_prod', 'method', 'batch',
                   'seq', 'unit', 'alpha', 'lr', 'steps', 'epochs')
    return {'runs': [{**item, **{key: configurations[run_id].get(key) for key in menu_fields},
                      'active': run_id == active, 'paused': run_id == paused,
                      'completed': run_id in completed,
                      'has_log': run_id in updated} for run_id, item in runs.items()],
            'run': run, 'config': config, 'status': status, 'rows': rows, 'log': tail,
            'updated': updated.get(selected),
            'best_epoch': result.get('best_epoch') if result else best['epoch'] if best else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--runs-root', type=Path, default=PROJECT / 'runs')
    args = parser.parse_args()
    root = args.runs_root.resolve()
    public_plan = read_json(PROJECT / 'experiments' / 'manifest.json')
    page = Path(__file__).with_name('training_monitor.html')

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            request = urlsplit(self.path)
            if request.path == '/':
                body, kind = page.read_bytes(), 'text/html; charset=utf-8'
            elif request.path == '/api/state':
                run = parse_qs(request.query).get('run', [''])[0]
                search_plan = read_json(root / 'hparam-search-plan.json')
                combination_plan = read_json(root / 'hparam-combination-plan.json')
                multiseed_plan = read_json(root / 'multiseed-plan.json')
                plan = {'runs': (search_plan.get('runs', []) + combination_plan.get('runs', [])
                                 + multiseed_plan.get('runs', [])
                                 + public_plan['runs']),
                        'configurations': {**public_plan['configurations'],
                                           **search_plan.get('configurations', {}),
                                           **combination_plan.get('configurations', {}),
                                           **multiseed_plan.get('configurations', {})}}
                search = read_json(root / 'hparam-search-status.json')
                combination = read_json(root / 'hparam-combination-status.json')
                multiseed = read_json(root / 'multiseed-status.json')
                state = snapshot(plan, root, run, multiseed or combination or search)
                state['search'] = search
                state['combination'] = combination
                state['multiseed'] = multiseed
                body = json.dumps(state, ensure_ascii=False, allow_nan=False).encode('utf-8')
                kind = 'application/json; charset=utf-8'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(('127.0.0.1', args.port), Handler) as server:
        print(f'http://127.0.0.1:{server.server_port}', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
