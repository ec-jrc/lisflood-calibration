import pytest

from liscal import schedulers


def function(value):
    return value


def test_dask():
    scheduler = schedulers.DaskScheduler(4)
    mapping = scheduler.create_mapping()
    futures = mapping(function, [2, 5])
    print(futures)
    values = scheduler.gather(futures)
    print(values)

    assert values == [2, 5]
    scheduler.close()


@pytest.mark.parametrize('scheduler', 
[
    schedulers.MultiprocessingScheduler(1),
    schedulers.MultiprocessingScheduler(4)
])
def test_mp(scheduler):
    mapping = scheduler.create_mapping()
    values = mapping(function, [2, 5])
    print(values)
    values = scheduler.gather(values)
    print(values)

    assert values == [2, 5]
    scheduler.close()
