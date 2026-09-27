import hashlib
import json
import pytest
from scripts.collect_m3w_easy_hurdle_training import validate


def fixture():
    manifest=dict(remote_path='/private',groups=[{'group':'g'}],held_rows_transferred=False)
    encoded=json.dumps({k:v for k,v in manifest.items() if k!='remote_path'},indent=2)+'\n'
    receipt=dict(manifest_sha256=hashlib.sha256(encoded.encode()).hexdigest(),held_outcomes_used=False,
        independent_roles_read=False,groups=1,pilot=True,complete=False,
        artifacts=[dict(group='g',arm=a) for a in ('marginal','supervised')])
    return dict(receipt=receipt,scheduler_state='COMPLETED|0:0',verified_checkpoint_files=2),manifest


def test_completed_pilot_receipt_matches_exact_manifest():
    r,m=fixture();assert validate(r,m,'pilot')==r['receipt']


@pytest.mark.parametrize('bad',['scheduler','held','hash','duplicate','wrong_phase'])
def test_refuses_incomplete_leaky_or_misaligned_receipts(bad):
    r,m=fixture();phase='pilot'
    if bad=='scheduler':r['scheduler_state']='RUNNING|0:0'
    if bad=='held':r['receipt']['held_outcomes_used']=True
    if bad=='hash':r['receipt']['manifest_sha256']='0'*64
    if bad=='duplicate':r['receipt']['artifacts'][1]=r['receipt']['artifacts'][0]
    if bad=='wrong_phase':phase='train'
    with pytest.raises(AssertionError):validate(r,m,phase)


def test_verification_receipt_requires_exact_replay_not_just_training():
    r,m=fixture()
    r['receipt'].update(groups=108,heads=216,model_updates=432000,
        paired_initialization_and_sample_chain_exact=True,all_checkpoints_finite_and_hash_verified=True,
        first_group_full_training_replay_exact_except_elapsed=True,replay_heads=2,replay_updates=4000,
        full_216head_retraining_replay=False,scientific_efficacy_evaluated=False)
    r['verified_checkpoint_files']=216
    validate(r,m,'verify')
    r['receipt']['first_group_full_training_replay_exact_except_elapsed']=False
    with pytest.raises(AssertionError):validate(r,m,'verify')
