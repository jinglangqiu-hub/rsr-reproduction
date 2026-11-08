"""Restore a paused RSR search after a shutdown, then continue its frozen queue."""

import argparse
import fcntl
import platform
import re
import subprocess
import sys
from time import monotonic

from run_search import (CACHE, FIRST_TARGET, PLAN, RUNS, STATE, TRAIN_END, VALID_END,
                        build_plan, digest, execute, read_json, save_json)


def checkpoint_history(log, epoch):
    lines = log.read_bytes().splitlines(keepends=True)
    marker = f'epoch: {epoch - 1} time:'.encode()
    end = next(i for i, line in enumerate(lines) if line.startswith(marker))
    loss = re.search(rb'Better valid loss: ([^\r\n]+)', lines[end - 1])
    if not loss:
        raise ValueError(f'Epoch {epoch} was not marked as a saved best checkpoint')
    return b''.join(lines[:end + 1]), float(loss[1])


def resume_run(plan, state, run, config, folder):
    import numpy as np
    from search_training import SearchRSR, graph, metrics_for, session_config, tf

    if platform.python_version() != plan['python'] or tf.__version__ != plan['tensorflow']:
        raise RuntimeError('Python or TensorFlow differs from the frozen environment')
    if not tf.config.list_physical_devices('GPU') or config['stage'] != 'relation':
        raise RuntimeError('Recovery requires an RSR relation-stage run and a registered GPU')
    market = config['market']
    preflight = read_json(CACHE / 'preflight.json')
    if digest(CACHE / f'{market}.npz') != preflight['markets'][market]['file_sha256']:
        raise ValueError(f'Prepared data changed: {market}')

    resume_epoch = state['pause']['checkpoint_epoch']
    if not 0 < resume_epoch < config['epochs']:
        raise ValueError(f'Invalid checkpoint epoch: {resume_epoch}')
    log = folder / 'train.log'
    history, saved_loss = checkpoint_history(log, resume_epoch)
    model = SearchRSR(config, run['seed'])
    inputs, (prediction, loss, regression, ranking), optimizer, _ = graph(model, False)
    seq = config['seq']
    offsets = np.arange(FIRST_TARGET - seq, TRAIN_END - seq)
    valid_offsets = range(TRAIN_END - seq, VALID_END - seq)
    rng = np.random.RandomState(run['seed'])
    for _ in range(resume_epoch):
        rng.shuffle(offsets)

    def validation(session):
        totals = np.zeros(3)
        pred, target, mask = (np.empty((model.batch_size, len(valid_offsets))) for _ in range(3))
        for i, offset in enumerate(valid_offsets):
            batch = model.get_batch(offset)
            value, *parts = session.run((prediction, loss, regression, ranking), dict(zip(inputs, batch)))
            totals += parts
            pred[:, i], target[:, i], mask[:, i] = value[:, 0], batch[3][:, 0], batch[1][:, 0]
        if not np.isfinite(totals).all():
            raise ValueError('Nonfinite validation loss')
        return totals / len(valid_offsets), (pred, target, mask)

    checkpoint = str(folder / 'best')
    saver = tf.train.Saver()
    with tf.Session(config=session_config()) as session:
        saver.restore(session, checkpoint)
        model.initialize_relation(session)
        restored_loss, best_predictions = validation(session)
        if not np.isclose(restored_loss[0], saved_loss, rtol=1e-5, atol=1e-10):
            raise ValueError(f'Checkpoint validation mismatch: {restored_loss[0]} vs {saved_loss}')
        best = {'manifest_sha256': digest(PLAN), 'id': run['id'], 'best_epoch': resume_epoch,
                'best_validation_loss': saved_loss, 'validation': metrics_for(*best_predictions)}
        print(f'Restored epoch {resume_epoch} with Adam state and verified validation loss: {saved_loss}',
              flush=True)

        with (folder / 'train.before-resume.log').open('xb') as original:
            original.write(log.read_bytes())
        temporary = folder / 'train.resume.tmp'
        temporary.write_bytes(history)
        temporary.replace(log)
        state['status'] = 'running'
        save_json(STATE, state)
        with log.open('a', encoding='utf-8', buffering=1) as output:
            print(f'Resumed from checkpoint after epoch {resume_epoch}', file=output)
            for epoch in range(resume_epoch, config['epochs']):
                start = monotonic()
                rng.shuffle(offsets)
                train_totals = np.zeros(3)
                for offset in offsets:
                    values = session.run((loss, regression, ranking, optimizer),
                                         dict(zip(inputs, model.get_batch(offset))))
                    train_totals += values[:3]
                train_means = train_totals / len(offsets)
                if not np.isfinite(train_means).all():
                    raise ValueError(f'Nonfinite training loss in epoch {epoch + 1}')
                valid_means, predictions = validation(session)
                metrics = metrics_for(*predictions)
                print('Train Loss:', *train_means, file=output)
                print('Valid MSE:', *valid_means, file=output)
                print('Valid performance:', {'mse': metrics['mse'], 'mrrt': metrics['mrr'],
                                             'btl': metrics['author_btl']}, file=output)
                if valid_means[0] < best['best_validation_loss']:
                    best.update(best_epoch=epoch + 1, best_validation_loss=float(valid_means[0]),
                                validation=metrics)
                    saver.save(session, checkpoint, write_meta_graph=False)
                    best_predictions = predictions
                    print('Better valid loss:', best['best_validation_loss'], file=output)
                print(f'epoch: {epoch} time: {monotonic() - start:.4f}', file=output)
    np.savez_compressed(folder / 'predictions.npz',
                        **dict(zip(('valid_pred', 'valid_gt', 'valid_mask'), best_predictions)))
    save_json(folder / 'result.json', best)


def verified_recovery():
    plan = read_json(PLAN)
    state = read_json(STATE)
    manifest_hash = digest(PLAN)
    if (state['status'] != 'paused' or state['plan_sha256'] != manifest_hash
            or read_json(CACHE / 'preflight.json')['plan_sha256'] != manifest_hash
            or plan != build_plan()):
        raise ValueError('No paused run matches the frozen and checked plan')
    run_id = state['active']
    run = plan['runs'][len(state['completed'])]
    if run['id'] != run_id or state['completed'] != [item['id'] for item in plan['runs'][:len(state['completed'])]]:
        raise ValueError('The paused task is not the next run in the frozen queue')
    config = plan['configurations'][run['configuration']]
    folder = RUNS / run_id
    if read_json(folder / 'config.json') != {'manifest_sha256': manifest_hash, 'run': run,
                                             'configuration': config}:
        raise ValueError(f'The paused configuration differs from the frozen plan: {run_id}')
    return plan, state, run, config, folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        resume_run(*verified_recovery())
        return
    with (RUNS / 'hparam-search.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan, state, run, _, folder = verified_recovery()
        try:
            with (folder / 'resume.log').open('ab') as log:
                subprocess.run([sys.executable, '-u', __file__, '--worker'],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError as error:
            current = read_json(STATE)
            current.update(status='stopped' if current['status'] == 'running' else 'paused',
                           message=f'Recovery failed for {run["id"]}. See {folder / "resume.log"}')
            save_json(STATE, current)
            raise RuntimeError(current['message']) from error
        execute(plan)


if __name__ == '__main__':
    main()
