import importlib.util
import sys
import types
import pathlib
import pytest


class DummyLogger:
    def __init__(self):
        self.logs = []

    def log(self, msg):
        self.logs.append(msg)


def load_position_manager():
    # Stub heavy optional dependencies so the module can be imported without them.
    stubs = [
        "aiohttp",
        "requests",
        "numpy",
        "pandas",
        "joblib",
    ]
    for name in stubs:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            if name == "pandas":
                mod.Series = type("Series", (), {})
                mod.DataFrame = type("DataFrame", (), {})
                mod.Timestamp = type("Timestamp", (), {})
                mod.Timedelta = type("Timedelta", (), {})
                mod.concat = lambda *a, **k: None
                mod.to_datetime = lambda *a, **k: None
                mod.isna = lambda *a, **k: False
            if name == "numpy":
                mod.std = lambda *a, **k: 0
                mod.mean = lambda *a, **k: 0
                mod.where = lambda *a, **k: 0
                mod.inf = float('inf')
                mod.isscalar = lambda x: not isinstance(x, (list, tuple, dict, set))
                mod.bool_ = bool
            sys.modules[name] = mod

    # websockets with exceptions
    if "websockets" not in sys.modules:
        ws_exc = types.ModuleType("websockets.exceptions")
        class DummyExc(Exception):
            pass
        ws_exc.ConnectionClosed = DummyExc
        ws_exc.ConnectionClosedOK = DummyExc
        ws_exc.ConnectionClosedError = DummyExc
        ws_mod = types.ModuleType("websockets")
        ws_mod.exceptions = ws_exc
        sys.modules["websockets"] = ws_mod
        sys.modules["websockets.exceptions"] = ws_exc

    # sklearn minimal stubs
    if "sklearn" not in sys.modules:
        skl = types.ModuleType("sklearn")
        ensemble = types.ModuleType("sklearn.ensemble")
        preprocessing = types.ModuleType("sklearn.preprocessing")
        model_selection = types.ModuleType("sklearn.model_selection")
        Dummy = type("Dummy", (), {})
        ensemble.RandomForestClassifier = Dummy
        preprocessing.StandardScaler = Dummy
        preprocessing.MinMaxScaler = Dummy
        preprocessing.LabelEncoder = Dummy
        model_selection.train_test_split = lambda *a, **k: None
        skl.ensemble = ensemble
        skl.preprocessing = preprocessing
        skl.model_selection = model_selection
        sys.modules["sklearn"] = skl
        sys.modules["sklearn.ensemble"] = ensemble
        sys.modules["sklearn.preprocessing"] = preprocessing
        sys.modules["sklearn.model_selection"] = model_selection

    # binance client stub
    if "binance" not in sys.modules:
        binance = types.ModuleType("binance")
        client_mod = types.ModuleType("binance.client")
        class DummyClient:
            pass
        client_mod.Client = DummyClient
        binance.client = client_mod
        sys.modules["binance"] = binance
        sys.modules["binance.client"] = client_mod

    root = pathlib.Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    path = root / "TraRyTrade_SelfSwimm_V1.py"
    spec = importlib.util.spec_from_file_location("TraRyTrade_SelfSwimm_V1", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def posmgr_cls():
    mod = load_position_manager()
    return mod.PositionManager


@pytest.fixture
def constants():
    mod = load_position_manager()
    return mod


def test_adjust_position_flip_long_to_short(posmgr_cls):
    pm = posmgr_cls(maker_fee=0.0, trade_unit_size=1.0, stop_loss_pct=0.1, logger=DummyLogger())
    pm.adjust_position(2, 100)
    pm.adjust_position(-3, 110)
    assert pm.position_units == -1
    assert pm.avg_entry_price == 110
    assert pm.realized_pnl == pytest.approx(20)
    assert pm.side == "SELLRUN"
    assert pm.stop_loss == pytest.approx(110 * 1.1)


def test_adjust_position_partial_close(posmgr_cls):
    pm = posmgr_cls(maker_fee=0.0, trade_unit_size=1.0, stop_loss_pct=0.1, logger=DummyLogger())
    pm.adjust_position(2, 100)
    pm.adjust_position(-1, 110)
    assert pm.position_units == 1
    assert pm.avg_entry_price == 100
    assert pm.realized_pnl == pytest.approx(10)
    assert pm.stop_loss == pytest.approx(100 * 0.9)


def test_get_unrealized_pnl_includes_fees(posmgr_cls):
    pm = posmgr_cls(maker_fee=0.01, trade_unit_size=1.0, stop_loss_pct=0.1, logger=DummyLogger())
    pm.adjust_position(1, 100)
    # open fee = 1 * 100 * 0.01 = 1
    pnl = pm.get_unrealized_pnl(110)
    assert pnl == pytest.approx(9)


def test_move_stop_to_breakeven_updates_stop_loss(posmgr_cls, constants):
    pm = posmgr_cls(maker_fee=0.0, trade_unit_size=1.0, stop_loss_pct=0.1, logger=DummyLogger())
    pm.adjust_position(1, 100)
    pm.move_stop_to_breakeven()
    assert pm.stop_loss == pytest.approx(100 * constants.move_stop_to_breakevenValveLONG)
    assert pm.stop_moved_to_breakeven

    pm.reset()
    pm.adjust_position(-1, 200)
    pm.move_stop_to_breakeven()
    assert pm.stop_loss == pytest.approx(200 * constants.move_stop_to_breakevenValveSHORT)
    assert pm.stop_moved_to_breakeven
