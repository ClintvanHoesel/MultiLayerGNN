import importlib
import inspect


def resolve_aggregator(name: str):
    """Import a ``torch_geometric.nn.aggr`` Aggregation class by name (on demand).

    Falls back to an explicit import of ``torch_geometric.nn.aggr.<name>`` so any
    aggregator shipped by PyG can be used without editing a registry.
    """
    try:
        return getattr(importlib.import_module("torch_geometric.nn.aggr"), name)
    except AttributeError:
        raise ValueError(
            f"unknown aggregation {name!r}: not found in torch_geometric.nn.aggr"
        ) from None


def build_aggregators(spec):
    """Instantiate one or several PyG aggregators from a config value.

    ``spec`` may be a single aggregator (class / instance / name), a
    ``"MeanAggregation+MaxAggregation"``-style string, or a list of any of
    those. Returns a list of instantiated aggregators; combine several with
    :class:`MultiAggregation` (see :class:`ScalarMoleculeModel`).
    """
    if isinstance(spec, str):
        items = [s.strip() for s in spec.split("+") if s.strip()]
    elif isinstance(spec, (list, tuple)):
        items = list(spec)
    else:
        items = [spec]
    aggs = []
    for item in items:
        if isinstance(item, str):
            aggs.append(resolve_aggregator(item)())
        elif inspect.isclass(item):
            aggs.append(item())
        else:
            aggs.append(item)
    if not aggs:
        raise ValueError("global_aggr must name at least one aggregation")
    return aggs
