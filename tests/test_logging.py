import logging

from clintrials.dosefinding import find_mtd
from clintrials.simulation import run_sims
from clintrials.logging import get_logger


def dummy_sim():
    return {"ok": True}


def test_find_mtd_logs(caplog):
    with caplog.at_level(logging.INFO):
        find_mtd(0.25, [0.15, 0.25, 0.35], verbose=True)
    assert any("MTD is" in record.message for record in caplog.records)


def test_run_sims_logs(caplog):
    with caplog.at_level(logging.INFO):
        run_sims(dummy_sim, n1=1, n2=1)
    assert any(record.levelno == logging.INFO for record in caplog.records)

def test_get_logger():
    logger = get_logger("test_logger")
    assert logger.name == "test_logger"
