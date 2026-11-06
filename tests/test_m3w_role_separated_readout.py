import sys
import pytest
from scripts import run_m3w_role_separated_readout as wrapper


def test_wrapper_creates_allocation_parent_without_changing_inference_args(tmp_path,monkeypatch):
    monkeypatch.setattr(wrapper.run,'PRIVATE',tmp_path/'private')
    args=['readout','infer','--resume'];monkeypatch.setattr(sys,'argv',args.copy())
    observed=[]
    def unchanged():
        assert (tmp_path/'private/allocation').is_dir()
        observed.append(sys.argv.copy())
    monkeypatch.setattr(wrapper.run,'main',unchanged)
    wrapper.main();assert observed==[args]


def test_wrapper_refuses_training(tmp_path,monkeypatch):
    monkeypatch.setattr(sys,'argv',['readout','train'])
    with pytest.raises(ValueError,match='cannot start'):wrapper.main()
