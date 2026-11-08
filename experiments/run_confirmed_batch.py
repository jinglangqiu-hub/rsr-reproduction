"""Run the eight approved public-seed configurations serially on the WSL GPU."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


PROJECT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT / 'experiments' / 'manifest.json'
WSL_PYTHON = '/home/luckythrower/.venvs/rsr-gpu/bin/python'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirmed-manifest-sha256', required=True)
    args = parser.parse_args()
    manifest_hash = digest(MANIFEST)
    if manifest_hash != args.confirmed_manifest_sha256.lower():
        raise SystemExit('Manifest differs from the confirmed batch')
    plan = json.loads(MANIFEST.read_text(encoding='utf-8'))
    configurations = plan['configurations']
    runs = plan['runs']
    expected = {(market, relation, method) for market in ('NASDAQ', 'NYSE')
                for relation in ('sector_industry', 'wikidata') for method in ('RSR_E', 'RSR_I')}
    observed = {(configurations[run['configuration']]['market'],
                 configurations[run['configuration']]['relation'],
                 configurations[run['configuration']]['method']) for run in runs}
    if len(runs) != 8 or observed != expected or any(run['seed'] != 123456789 for run in runs):
        raise SystemExit('The confirmed grid must contain eight distinct runs with seed 123456789')

    log_directory = PROJECT / 'audit' / f'public-seed-batch-{manifest_hash[:12]}'
    log_directory.mkdir(exist_ok=False)
    state_path = log_directory / 'status.json'
    records = [{'id': run['id'], 'status': 'queued'} for run in runs]
    state = {'manifest_sha256': manifest_hash, 'status': 'running', 'runs': records}

    def save_state():
        temporary = state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
        temporary.replace(state_path)

    wsl = ['wsl.exe', '-d', 'Ubuntu', '--cd', str(PROJECT), '--exec', WSL_PYTHON]
    save_state()
    try:
        for run, record in zip(runs, records):
            if digest(MANIFEST) != manifest_hash:
                raise RuntimeError('Manifest changed during the confirmed batch')
            config = configurations[run['configuration']]
            command = [*wsl, 'environment/run_gpu.py', 'compat/run_experiment.py',
                       '--run', run['id'], '--execute']
            if config['parameter_provenance'] == 'market_template':
                command.append('--allow-market-template')
            log_path = log_directory / f'{run["id"]}.log'
            record.update(status='running', console_log=log_path.name)
            save_state()
            print(f'Starting {run["id"]}', flush=True)
            with log_path.open('xb') as log:
                outcome = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if outcome.returncode:
                record.update(status='stopped', exit_code=outcome.returncode)
                raise RuntimeError(f'{run["id"]} exited with {outcome.returncode}')
            result_path = PROJECT / 'runs' / run['id'] / 'result.json'
            result = json.loads(result_path.read_text(encoding='utf-8'))
            if result.get('manifest_sha256') != manifest_hash or result.get('id') != run['id']:
                raise RuntimeError(f'Unexpected result for {run["id"]}')
            record.update(status='completed', result=result_path.relative_to(PROJECT).as_posix())
            save_state()
        state['status'] = 'completed'
        save_state()
    except (OSError, ValueError, RuntimeError) as error:
        state.update(status='stopped', message=str(error))
        save_state()
        raise


if __name__ == '__main__':
    main()
