"""Global receiver-attention mass control at exactly the frozen T bank/budget."""
import torch
from block_selector import components


def select(a, v, labels, protected, candidates, mandatory, budget, mode):
    assert mode == 'G'
    assert a.ndim == v.ndim == 3 and a.shape[0] == v.shape[0]
    assert a.shape[-1] == v.shape[1] == labels.shape[-1]
    assert not bool((protected & candidates).any())
    assert bool((protected | candidates).all())
    assert not bool((mandatory & ~candidates).any())
    assert bool((mandatory.sum(-1) == mandatory.sum(-1)[0]).all())
    assert bool((mandatory.sum(-1) <= budget).all())
    assert bool((candidates.sum(-1) >= budget).all())
    a, v = a.double(), v.double()
    ranks = torch.argsort(a.sum(1).masked_fill(~(candidates & ~mandatory), -torch.inf),
                         dim=-1, descending=True, stable=True)
    bank_n = min(2 * budget, int(candidates.sum(-1).min()))
    bank_extra = bank_n - int(mandatory.sum(-1)[0])
    bank = mandatory.clone().scatter(1, ranks[:, :bank_extra], True)
    free_n = budget - int(mandatory.sum(-1)[0])
    selected = protected | mandatory
    selected.scatter_(1, ranks[:, :free_n], True)
    assert not bool((selected & candidates & ~bank).any())
    assert bool(((selected & candidates).sum(-1) == budget).all())
    # Diagnostics use V after selection; V does not influence the ranking.
    full = torch.ones_like(selected)
    target = components(a, v, labels, full)
    observed = components(a, v, labels, selected)
    report = dict(mode='G', bank_rows=bank_n, prompt_budget=budget,
                  total_mse=float((target.sum(1)-observed.sum(1)).square().mean()),
                  true_blocks_mse=float((target-observed).square().mean()),
                  objective='sum_attention_mass_same_64_queries',
                  shuffle_changed_fraction=0.0)
    return selected, report
