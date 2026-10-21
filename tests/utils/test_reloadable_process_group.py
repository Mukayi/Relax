#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""ReloadableProcessGroup must forward what torch 2.13 added to ProcessGroup.

``ReloadableProcessGroup`` subclasses the C++ ``ProcessGroup``, so a method or
property it does not override resolves to the base class and acts on the
wrapper's own, backend-less ProcessGroup instead of the inner group: the
collectives segfault on ``wait()`` and the property accessors raise. Comparing
the ``ProcessGroup`` bindings of torch v2.9.0 and v2.13.0
(``torch/csrc/distributed/c10d/init.cpp``) gives exactly the five collectives
and the one property below; nothing was removed.
"""

import os

import pytest
import torch.distributed as dist

from relax.utils.reloadable_process_group import ReloadableProcessGroup


ADDED_COLLECTIVES = (
    "reduce_scatter_single",
    "reduce_scatter_single_coalesced",
    "all_gather_single",
    "all_gather_single_coalesced",
    "all_to_all_single",
)


class _InnerGroup:
    """Records the calls that reach it."""

    def __init__(self):
        self.calls = []
        self.use_pg_for_symm_mem_rendezvous = False

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return name

        return call


@pytest.fixture
def wrapped(monkeypatch):
    monkeypatch.setattr(dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 2)
    inner = _InnerGroup()
    group = ReloadableProcessGroup(inner, [0, 1])
    yield group, inner
    ReloadableProcessGroup.GROUPS[os.getpid()].remove(group)


@pytest.mark.parametrize("name", ADDED_COLLECTIVES)
def test_collectives_added_in_torch_2_13_reach_the_inner_group(wrapped, name):
    group, inner = wrapped
    assert getattr(group, name)("output", "input", opts="opts") == name
    assert inner.calls == [(name, ("output", "input"), {"opts": "opts"})]


def test_symm_mem_rendezvous_flag_is_the_inner_groups(wrapped):
    group, inner = wrapped
    group.use_pg_for_symm_mem_rendezvous = True
    assert inner.use_pg_for_symm_mem_rendezvous is True
    assert group.use_pg_for_symm_mem_rendezvous is True
