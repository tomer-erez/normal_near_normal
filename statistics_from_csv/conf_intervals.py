import numpy as np

def bootstrap_ci(results, n_boot=10000, ci=95, seed=42):
    """
    results: list/array of 0/1 values, one per query
    n_boot: number of bootstrap samples
    ci: confidence interval percentage

    Returns:
        mean_score
        lower_bound
        upper_bound
    """
    rng = np.random.default_rng(seed)

    results = np.asarray(results)
    n = len(results)

    bootstrap_scores = []

    for _ in range(n_boot):
        sample = rng.choice(results, size=n, replace=True)
        bootstrap_scores.append(sample.mean())

    alpha = (100 - ci) / 2

    lower = np.percentile(bootstrap_scores, alpha)
    upper = np.percentile(bootstrap_scores, 100 - alpha)

    return results.mean(), lower, upper


# Example
results = [0 for i in range(400)]+[1 for j in range(600)]
print(f"p@1 acc = {sum(results)/len(results)}")
mean_score, lower, upper = bootstrap_ci(results)

print(f"P@1 = {mean_score:.3f}")
print(f"95% CI = [{lower:.3f}, {upper:.3f}]")