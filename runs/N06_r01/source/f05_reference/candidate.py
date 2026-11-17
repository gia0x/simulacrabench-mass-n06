"""F05: one L2-regularized categorical logistic regression per survey question."""
import numpy as np

from . import forest_main as base


def _features(codes, widths, context):
    """Sparse GIVEN indicators plus the one prespecified survey context pair."""
    from scipy.sparse import csr_matrix

    rows = len(codes)
    if not widths:
        return csr_matrix((rows, 0), dtype=np.float64)
    offsets = np.cumsum([0] + list(widths[:-1]))
    columns = codes + offsets
    column_count = sum(widths)
    if context:
        first, second = context
        crossed = codes[:, first] * widths[second] + codes[:, second]
        columns = np.column_stack((columns, column_count + crossed))
        column_count += widths[first] * widths[second]
    indices = columns.ravel()
    indptr = np.arange(rows + 1) * columns.shape[1]
    return csr_matrix((np.ones(len(indices)), indices, indptr), shape=(rows, column_count))


def fit_predict(x_fit, y_fit, x_query, given, targets, widths_x, widths_y, cfg, deadline):
    """Return complete per-item supports; only labels from the supplied fit rows enter training."""
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits

    base._check_budget(deadline)
    context = base._context_columns(given)
    train = _features(x_fit, widths_x, context)
    query = _features(x_query, widths_x, context)
    result = []
    with threadpool_limits(limits=1):
        for j, width in enumerate(widths_y):
            base._check_budget(deadline)
            valid = y_fit[:, j] >= 0
            labels = y_fit[valid, j]
            prior = np.bincount(labels, minlength=width).astype(float) + cfg['marginal_pseudocount']
            prior /= prior.sum()
            if not train.shape[1] or len(np.unique(labels)) < 2:
                result.append(np.tile(prior, (len(x_query), 1)))
                continue
            model = LogisticRegression(C=cfg['logistic_c'], solver='lbfgs', max_iter=cfg['logistic_max_iter'], tol=1e-3)
            model.fit(train[valid], labels)
            base._check_budget(deadline)
            p = np.zeros((len(x_query), width))
            p[:, model.classes_.astype(int)] = model.predict_proba(query)
            result.append((1 - cfg['prior_share']) * p + cfg['prior_share'] * prior)
    base._check_budget(deadline)
    return result
