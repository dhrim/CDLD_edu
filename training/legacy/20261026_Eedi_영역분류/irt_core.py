from pathlib import Path
import time,gc,json,os
import numpy as np
from scipy import sparse
from scipy.special import expit,logsumexp
RESUME=True
def save_npz(path, **values):
    path = Path(path)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **values)
    temporary.replace(path)

def irt_grid(nodes):
    theta = np.linspace(-4.0, 4.0, nodes)
    prior = np.exp(-0.5 * theta ** 2)
    return theta, prior / prior.sum()

def sparse_responses(data, rows, n_people, n_items):
    people = data["person"][rows]
    items = data["item"][rows]
    observed = sparse.csr_matrix((np.ones(len(rows)), (people, items)), shape=(n_people, n_items))
    correct = sparse.csr_matrix((data["correct"][rows].astype(float), (people, items)), shape=(n_people, n_items))
    if observed.data.max() != 1:
        raise ValueError("Repeated person–item pairs must be resolved before fitting.")
    return observed, correct

def irt_expectation(observed, correct, a, b, theta, prior, person_batch):
    probability = np.clip(expit(a[:, None] * (theta[None, :] - b[:, None])), 1e-9, 1 - 1e-9)
    log_odds = np.log(probability) - np.log1p(-probability)
    log_incorrect = np.log1p(-probability)
    expected_trials = np.zeros_like(probability)
    expected_correct = np.zeros_like(probability)
    likelihood, last_message = 0.0, time.monotonic()
    for start in range(0, observed.shape[0], person_batch):
        obs = observed[start:start + person_batch]
        yes = correct[start:start + person_batch]
        log_posterior = yes @ log_odds + obs @ log_incorrect + np.log(prior)[None, :]
        normalizer = logsumexp(log_posterior, axis=1)
        posterior = np.exp(log_posterior - normalizer[:, None])
        likelihood += float(normalizer.sum())
        expected_trials += obs.T @ posterior
        expected_correct += yes.T @ posterior
        if time.monotonic() - last_message > 30:
            print(f"  IRT expectation: {min(start + person_batch, observed.shape[0]):,}/{observed.shape[0]:,} people", flush=True)
            last_message = time.monotonic()
    return expected_trials, expected_correct, likelihood

def fit_irt(data, rows, description, config, model, checkpoint):
    observed, correct = sparse_responses(data, rows, description["people"], description["items"])
    theta, prior = irt_grid(config["irt_nodes"])
    trials = np.asarray(observed.sum(axis=0)).ravel()
    successes = np.asarray(correct.sum(axis=0)).ravel()
    active = trials > 0
    initial_rate = np.clip(np.divide(successes, trials, out=np.full_like(trials, 0.5), where=active), .01, .99)
    a, b = np.ones(len(trials)), -np.log(initial_rate / (1 - initial_rate))
    first_iteration, history = 0, []
    checkpoint = Path(checkpoint)
    if RESUME and checkpoint.exists():
        saved = np.load(checkpoint)
        a, b = saved["a"], saved["b"]
        first_iteration = int(saved["iteration"])
        history = saved["history"].tolist()
        print(f"Resuming {model} at iteration {first_iteration}")
    converged = bool(history and history[-1][1] < config["irt_tolerance"])
    iteration = first_iteration - 1
    for iteration in range(first_iteration, config["irt_max_iterations"]):
        if converged:
            iteration = first_iteration - 1
            break
        expected_trials, expected_correct, likelihood = irt_expectation(
            observed, correct, a, b, theta, prior, config["irt_person_batch"])
        old_a, old_b = a.copy(), b.copy()
        for step in range(25):
            probability = np.clip(expit(a[:, None] * (theta[None, :] - b[:, None])), 1e-9, 1 - 1e-9)
            weight = expected_trials * probability * (1 - probability)
            residual = expected_correct - expected_trials * probability
            gradient_b = (-a[:, None] * residual).sum(axis=1)
            hessian_b = -(a[:, None] ** 2 * weight).sum(axis=1)
            b -= gradient_b / np.minimum(hessian_b, -1e-9)
            if model == "2PL":
                probability = np.clip(expit(a[:, None] * (theta[None, :] - b[:, None])), 1e-9, 1 - 1e-9)
                weight = expected_trials * probability * (1 - probability)
                residual = expected_correct - expected_trials * probability
                distance = theta[None, :] - b[:, None]
                gradient_a = (residual * distance).sum(axis=1)
                hessian_a = -(weight * distance ** 2).sum(axis=1)
                a = np.clip(a - gradient_a / np.minimum(hessian_a, -1e-9), .1, 4.0)
        change = float(max(np.max(abs(a - old_a)), np.max(abs(b - old_b))))
        if not np.isfinite(change):
            raise FloatingPointError(f"Nonfinite {model} update")
        history.append([iteration + 1, change, likelihood])
        converged = change < config["irt_tolerance"]
        if iteration == 0 or (iteration + 1) % 10 == 0 or converged:
            print(f"{model}: iteration {iteration+1}, max parameter change={change:.3g}", flush=True)
            save_npz(checkpoint, a=a, b=b, iteration=iteration + 1, history=np.asarray(history))
        if converged:
            break
    del observed, correct
    gc.collect()
    return {"a": a, "b": b, "active": active, "converged": converged,
            "iterations": iteration + 1, "history": np.asarray(history),
            "slope_boundary_items": int((((a <= .1) | (a >= 4.0)) & active).sum())}