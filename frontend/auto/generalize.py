"""Lossless expansion of the native GEMM template into general task syntax.

This changes representation only. It preserves native component accounting and
iteration overlap, including DRAM request order; it does not add transfers,
barriers, output materialization or timing formulas. New schedules should be
checked against this reference before attributing differences to search space.
"""
from copy import deepcopy


COMPONENTS = ('matrix', 'vector', 'buffer_load', 'buffer_store', 'dram', 'noc_tx', 'noc_rx')


def expand_gemm(operator, tensors):
    if operator['type'] != 'gemm':
        raise ValueError('Only the native GEMM template has been audited')
    count = operator['iteration']
    if count <= 0:
        raise ValueError('Positive iteration count required')
    execution = operator['execution']
    if set(execution) - set(COMPONENTS) or execution.get('noc_tx') or execution.get('noc_rx'):
        raise ValueError('Unexpected native component or communication task')
    raw = [{key: deepcopy(execution.get(key, [])) if key not in ('dram', 'noc_tx', 'noc_rx') else []
            for key in COMPONENTS} for _ in range(count)]
    by_name = {t['name']: t for t in tensors}
    # The native C++ GEMM path inserts matrix/vector/buffer at sim iter r+1.
    # The general path does the same; reads occur at raw r, writes at r+2.
    for task in execution['dram']:
        shape = by_name[task['name']]['shape']
        base = list(task['access_base'])
        accesses = 0
        for iteration in range(task['init_iter'], count+2, task['stride_iter']):
            slot = iteration - (2 if task['is_write'] else 0)
            if not 0 <= slot < count:
                raise ValueError('DRAM event cannot be represented at the same native iteration')
            item = deepcopy(task)
            item.update(access_base=list(base), access_stride_add=[1]*len(base),
                        access_offset_add=[0]*len(base), init_iter=0, stride_iter=1, total_iter=1)
            raw[slot]['dram'].append(item)
            accesses += 1
            if accesses >= task['total_iter']:
                break
            for d, stride in enumerate(task['access_stride_add']):
                if stride > 0 and accesses % stride == 0:
                    base[d] = (base[d]+task['access_offset_add'][d]) % shape[d]
    return dict(iteration=count, execution=raw)
