"""Check the adapted model graphs on GPU without creating an optimizer."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit, softmax

sys.path.insert(0, str(Path(__file__).resolve().parent / 'training'))
from rank_lstm import RankLSTM, tf
from relation_rank_lstm import ReRaLSTM


def forward(model, feature, mask, price, target, *, trace=False):
    inputs, valid, base, labels, prediction, loss, mse, rank, *extra = model.build_graph()
    assert extra[-1] is None
    config = tf.ConfigProto(allow_soft_placement=False)
    config.gpu_options.allow_growth = True
    metadata = tf.RunMetadata()
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE) if trace else None
    variables = tf.trainable_variables()
    with tf.Session(config=config) as session:
        session.run(tf.global_variables_initializer())
        if isinstance(model, ReRaLSTM):
            model.initialize_relation(session)
        result = session.run(
            [prediction, loss, mse, rank, *extra[:-1]],
            {inputs: feature, valid: mask, base: price, labels: target},
            options=options, run_metadata=metadata,
        )
        weights = dict(zip((v.name for v in variables), session.run(variables)))
    devices = sorted({stats.device for stats in metadata.step_stats.dev_stats
                      if stats.node_stats and '/stream:' in stats.device and 'GPU:0' in stats.device})
    if trace:
        assert devices, 'Expected GPU kernel execution'
    assert all(np.isfinite(value).all() for value in result)
    return result, weights, devices


def leaky(x):
    return np.maximum(x, 0.2 * x)


def dense(x, weights, name):
    return leaky(x @ weights[f'{name}/kernel:0'] + weights[f'{name}/bias:0'])


def check_synthetic():
    rng = np.random.default_rng(13)
    stocks, steps, units = 5, 4, 3
    market = rng.normal(0, .3, (stocks, steps, 5)).astype('float32')
    embedding = rng.normal(0, .3, (stocks, units)).astype('float32')
    relations = rng.integers(0, 2, (stocks, stocks, 2)).astype('float32')
    relations[np.arange(stocks), np.arange(stocks), 0] = 1
    bias = np.where(relations.any(axis=2), 0., -1e9).astype('float32')
    mask = np.array([[1], [0], [1], [1], [1]], dtype='float32')
    price = np.linspace(.7, 1.3, stocks, dtype='float32')[:, None]
    target = rng.normal(0, .05, (stocks, 1)).astype('float32')
    results = {}
    for name in ('Rank_LSTM', 'RSR_I', 'RSR_E'):
        model = object.__new__(RankLSTM if name == 'Rank_LSTM' else ReRaLSTM)
        model.batch_size, model.gpu = stocks, True
        model.parameters = {'seq': steps, 'unit': units, 'alpha': .1}
        if name == 'Rank_LSTM':
            model.fea_dim = market.shape[-1]
            result, weights, devices = forward(model, market, mask, price, target, trace=True)
            kernel = weights['rnn/basic_lstm_cell/kernel:0']
            offset = weights['rnn/basic_lstm_cell/bias:0']
            cell = hidden = np.zeros((stocks, units), dtype='float32')
            for step in market.transpose(1, 0, 2):
                i, j, f, o = np.split(np.concatenate([step, hidden], axis=1) @ kernel + offset, 4, axis=1)
                cell = cell * expit(f + 1.) + expit(i) * np.tanh(j)
                hidden = np.tanh(cell) * expit(o)
            np.testing.assert_allclose(result[4], hidden, rtol=2e-5, atol=1e-6)
            features = hidden
        else:
            model.seed = 123456789
            model.rel_encoding, model.rel_mask = relations, bias
            model.flat = False
            model.inner_prod = name == 'RSR_E'
            result, weights, devices = forward(model, embedding, mask, price, target, trace=True)
            strength = dense(relations, weights, 'dense')[:, :, 0]
            if model.inner_prod:
                strength *= embedding @ embedding.T
            else:
                strength += dense(embedding, weights, 'dense_1') + dense(embedding, weights, 'dense_2').T
            propagated = softmax(strength + bias, axis=0) @ embedding
            features = np.concatenate([embedding, propagated], axis=1)
        returns = (dense(features, weights, 'reg_fc') - price) / price
        mse = np.sum(np.square(returns - target) * mask) / mask.sum()
        pairwise = (returns - returns.T) * (target.T - target) * (mask @ mask.T)
        rank = np.maximum(pairwise, 0.).mean()
        for actual, expected in zip(result[:4], [returns, mse + .1 * rank, mse, rank]):
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-6)
        results[name] = {'gpu_streams': devices, 'loss': float(result[1])}
    return results


def check_author_data(market):
    market_settings = {
        'NASDAQ': ('wikidata', 16, 64, .1, 2),
        'NYSE': ('sector_industry', 8, 32, 10., 0),
    }
    relation, steps, units, alpha, embedding_index = market_settings[market]
    data = Path(__file__).resolve().parent / 'data' / '2013-01-01'
    tickers = f'{market}_tickers_qualify_dr-0.98_min-5_smooth.csv'
    embedding = f'{market}_rank_lstm_seq-{steps}_unit-{units}_{embedding_index}.csv.npy'
    model = ReRaLSTM(str(data), market, tickers, relation, embedding,
                    {'seq': steps, 'unit': units, 'alpha': alpha}, gpu=True)
    offset = model.valid_index - steps - model.steps + 1
    feature, mask, price, target = model.get_batch(offset)
    result, _, devices = forward(model, feature, mask, price, target, trace=True)
    assert result[0].shape == (len(model.tickers), 1)
    return {'market': market, 'stocks': len(model.tickers), 'relation': relation,
            'embedding': embedding, 'validation_offset': offset,
            'valid_stocks': int(mask.sum()), 'gpu_streams': devices,
            'prediction_shape': result[0].shape, 'finite': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--market', choices=['NASDAQ', 'NYSE'])
    args = parser.parse_args()
    report = {'optimizer_steps': 0}
    if args.market:
        report['author_data_forward'] = check_author_data(args.market)
    else:
        report['synthetic_numpy_agreement'] = check_synthetic()
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
