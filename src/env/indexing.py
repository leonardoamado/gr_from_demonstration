import collections
import itertools
import operator
from typing import Collection, Dict, Iterable, Tuple, Type, TypeVar, Protocol, ClassVar

import numpy as np

# from pddlenv.base import ArityObject, PDDLObject
from pddlgym.structs import Predicate, Type, TypedEntity

LiteralIndices = Dict[int, Tuple[np.array, ...]]
LiteralShapes = Dict[int, Tuple[int, ...]]


class ArityObject(Protocol):
    arity: ClassVar[int]


P = TypeVar("P", bound=ArityObject)


def _grounded_literal_index(literal, sorted_objects, sorted_predicates):
    index = tuple(sorted_objects[o] for o in literal.objects)
    return index + (sorted_predicates[type(literal)],)


def _shape_from_grouped_predicates(num_objects: int,
                                   grouped_pred: Iterable[Tuple[int, Collection]]
                                   ) -> LiteralShapes:
    return {
        arity: (num_objects,) * arity + (len(preds),)
        for arity, preds in grouped_pred
    }


def compute_shapes(num_objects: int,
                   predicates: Collection[Predicate]) -> LiteralShapes:
    grouped_pred = itertools.groupby(sorted(predicates, key=operator.attrgetter("arity")),
                                     key=operator.attrgetter("arity"))
    return _shape_from_grouped_predicates(
        num_objects,
        grouped_pred=((arity, tuple(preds)) for arity, preds in grouped_pred),
    )


def compute_indices(literals: Iterable,  # literals: Iterable[Collection[P]],
                    objects: Collection[TypedEntity],
                    predicates: Collection) -> Tuple[LiteralIndices, LiteralShapes]:
    grouped_pred = itertools.groupby(sorted(predicates, key=operator.attrgetter("arity")),
                                     key=operator.attrgetter("arity"))
    sorted_pred = {
        arity: {p: i for i, p in enumerate(sorted(preds, key=operator.attrgetter("name")))}
        for arity, preds in grouped_pred
    }

    objects = {o: i for i, o in enumerate(sorted(objects))}

    indices = collections.defaultdict(list)
    for i, lits in enumerate(literals):
        for lit in lits:
            arity = lit.arity
            indices[arity].append((i,) + _grounded_literal_index(lit, objects, sorted_pred[arity]))

    shapes = _shape_from_grouped_predicates(len(objects), sorted_pred.items())
    tupled_indices = {
        k: tuple(np.array(i) for i in zip(*idx))
        for k, idx in indices.items()
    }

    return tupled_indices, shapes


def ravel_literal_indices(indices: LiteralIndices,
                          shapes: LiteralShapes) -> Tuple[np.ndarray, np.ndarray]:
    arity_offsets = dict(zip(
        shapes.keys(),
        np.cumsum([0] + [np.prod(shape) for shape in list(shapes.values())[:-1]])
    ))
    batch_idx, flat_idx = zip(*(
        (idx[0], np.ravel_multi_index(idx[1:], shapes[arity]) + arity_offsets[arity])
        for arity, idx in indices.items()
    ))
    return np.concatenate(batch_idx), np.concatenate(flat_idx)


def _unravel_index(indices, arity_offset, shape):
    batch_idx, idx = zip(*indices)
    return (np.array(batch_idx),) + np.unravel_index(np.subtract(idx, arity_offset), shape)


def unravel_literal_indices(indices: Tuple[np.ndarray, np.ndarray],
                            shapes: LiteralShapes) -> LiteralIndices:
    arity_offsets = np.cumsum([np.prod(shape) for shape in shapes.values()])
    arity_indices = np.digitize(indices[1], arity_offsets)
    arities = np.take(list(shapes.keys()), arity_indices)

    ravelled = collections.defaultdict(list)
    for arity, batch_idx, idx in zip(arities, *indices):
        ravelled[arity].append((batch_idx, idx))

    return {
        arity: _unravel_index(ravelled[arity], offset, shapes[arity])
        for arity, offset in zip(shapes, np.pad(arity_offsets[:-1], (1, 0)))
        if arity in ravelled
    }