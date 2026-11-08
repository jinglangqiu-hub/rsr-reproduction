"""Stage and audit author artifacts. Run under Python 3.12+ with NumPy."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'upstream' / 'data'
DATA = ROOT / 'compat' / 'data'
PRETRAIN = ROOT / 'downloads' / 'pretrain.tar.gz'
MARKETS = {'NASDAQ': 1026, 'NYSE': 1737}


def sha256(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def stage_archive(archive, subtree):
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if (not path.parts or path.parts[0] != subtree or '..' in path.parts
                    or '\\' in member.name or ':' in member.name
                    or not (member.isdir() or member.isfile())):
                raise ValueError(f'Unexpected archive member: {member.name}')
        for member in members:
            destination = DATA.joinpath(*PurePosixPath(member.name).parts)
            if member.isdir():
                destination.mkdir(exist_ok=True, parents=True)
                continue
            if destination.is_file() and destination.stat().st_size == member.size:
                continue
            destination.parent.mkdir(exist_ok=True, parents=True)
            partial = destination.with_suffix(destination.suffix + '.part')
            with tar.extractfile(member) as source, partial.open('wb') as output:
                shutil.copyfileobj(source, output, 1024 * 1024)
            if partial.stat().st_size != member.size:
                raise ValueError(f'Incomplete archive member: {member.name}')
            partial.replace(destination)


def audit_market(market, stock_count):
    ticker_file = next(UPSTREAM.glob(f'{market}_tickers_qualify_*.csv'))
    calendar_file = UPSTREAM / f'{market}_aver_line_dates.csv'
    for source in (ticker_file, calendar_file):
        shutil.copyfile(source, DATA / source.name)
    tickers = ticker_file.read_text().splitlines()
    assert len(tickers) == len(set(tickers)) == stock_count
    calendar = calendar_file.read_text().splitlines()[29:]
    used_calendar = calendar[:-1] if market == 'NASDAQ' else calendar
    assert len(used_calendar) == 1245
    missing = 0
    source_files = hashlib.sha256()
    for ticker in tickers:
        name = f'{market}_{ticker}_1.csv'
        source = UPSTREAM / '2013-01-01' / name
        destination = DATA / '2013-01-01' / name
        raw = source.read_bytes()
        source_files.update(f'{name} {hashlib.sha256(raw).hexdigest()}\n'.encode())
        values = np.loadtxt(source, delimiter=',', dtype=np.float32)
        assert values.shape == (len(calendar), 6), (name, values.shape)
        np.testing.assert_array_equal(values[:, 0], np.arange(len(calendar)))
        assert np.isfinite(values).all(), name
        selected = values[:len(used_calendar), -1]
        missing += int(np.sum(selected == -1234))
        assert np.all((selected == -1234) | (selected > 0)), name
        if not destination.exists() or destination.read_bytes() != raw:
            destination.write_bytes(raw)
    relations = {}
    for directory, suffix in [('sector_industry', 'industry'), ('wikidata', 'wiki')]:
        path = DATA / 'relation' / directory / f'{market}_{suffix}_relation.npy'
        matrix = np.load(path, mmap_mode='r', allow_pickle=False)
        assert matrix.ndim == 3 and matrix.shape[:2] == (stock_count, stock_count)
        off_diagonal_pairs = 0
        active_channels = np.zeros(matrix.shape[2], dtype=bool)
        self_edges = 0
        for offset in range(0, stock_count, 32):
            block = matrix[offset:offset + 32]
            assert block.min() == 0 and block.max() <= 1, path
            active_channels |= np.any(block, axis=(0, 1))
            connected = np.any(block, axis=2)
            rows = np.arange(len(block))
            diagonal = int(connected[rows, offset + rows].sum())
            self_edges += diagonal
            off_diagonal_pairs += int(connected.sum()) - diagonal
        relations[directory] = {
            'shape': matrix.shape, 'dtype': str(matrix.dtype), 'sha256': sha256(path),
            'active_channels_including_self': int(active_channels.sum()),
            'self_edges': self_edges,
            'off_diagonal_density': off_diagonal_pairs / (stock_count * (stock_count - 1)),
        }
        if directory == 'sector_industry':
            mapping = json.loads(path.with_name(f'{market}_industry_ticker.json').read_text())
            indices = {ticker: index for index, ticker in enumerate(tickers)}
            groups = [members for members in mapping.values() if len(members) > 1]
            assert matrix.shape[2] == len(groups) + 1
            rows = np.arange(stock_count)
            assert np.all(matrix[rows, rows, -1] == 1)
            for channel, group in enumerate(groups):
                rows = np.array([indices[ticker] for ticker in group])
                assert np.all(matrix[rows[:, None], rows[None, :], channel] == 1), (market, channel)
            relations[directory].update({
                'ticker_axis_checked_against_industry_groups': True,
                'json_industry_groups': len(mapping), 'non_singleton_groups': len(groups),
            })
    return {
        'stocks': stock_count, 'source_rows': len(calendar), 'used_rows': len(used_calendar),
        'ticker_sha256': sha256(ticker_file), 'eod_manifest_sha256': source_files.hexdigest(),
        'missing_close_cells': missing, 'relation_arrays': relations,
        'split_dates': {key: [used_calendar[start][:10], used_calendar[end - 1][:10], end - start]
                        for key, start, end in [('train', 0, 756), ('valid', 756, 1008), ('test', 1008, 1245)]},
    }


def audit_pretrain():
    if not PRETRAIN.is_file():
        return {'status': 'archive_not_yet_downloaded'}
    stage_archive(PRETRAIN, 'pretrain')
    arrays = {}
    for path in sorted((DATA / 'pretrain').glob('*.npy')):
        matrix = np.load(path, mmap_mode='r', allow_pickle=False)
        market = path.name.split('_')[0]
        assert matrix.ndim == 3 and matrix.shape[0] == MARKETS[market], path
        for offset in range(0, matrix.shape[0], 32):
            assert np.isfinite(matrix[offset:offset + 32]).all(), path
        arrays[path.name] = {'shape': matrix.shape, 'dtype': str(matrix.dtype), 'sha256': sha256(path)}
    for name, stocks, seq, units in [
        ('NASDAQ_rank_lstm_seq-16_unit-64_2.csv.npy', 1026, 16, 64),
        ('NYSE_rank_lstm_seq-8_unit-32_0.csv.npy', 1737, 8, 32),
    ]:
        shape = arrays[name]['shape']
        assert shape[0] == stocks and shape[1] >= 1245 - seq and shape[2] == units, (name, shape)
    return {'status': 'verified', 'archive_sha256': sha256(PRETRAIN), 'arrays': arrays}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain-only', action='store_true')
    args = parser.parse_args()
    output = ROOT / 'audit' / 'author-data-verification.json'
    if args.pretrain_only:
        report = json.loads(output.read_text(encoding='utf-8'))
    else:
        (DATA / '2013-01-01').mkdir(parents=True, exist_ok=True)
        relation_archive = UPSTREAM / 'relation.tar.gz'
        stage_archive(relation_archive, 'relation')
        report = {'upstream_relation_archive_sha256': sha256(relation_archive),
                  'markets': {name: audit_market(name, count) for name, count in MARKETS.items()}}
    report['pretrain'] = audit_pretrain()
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(output)
