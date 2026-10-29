from scripts.check_m3w_create_queue_verified import classify_response


def test_login_banner_and_zero_exit_are_not_verified_queue():
    value=classify_response(dict(returncode=0,stdout='',stderr='Authenticate via portal'),'abc')
    assert not value['queue_snapshot_verified']
    assert value['queue_lines'] is None and not value['job_completion_inferred']


def test_marked_empty_queue_is_verified_snapshot_not_job_completion():
    value=classify_response(dict(returncode=0,stdout='M3W_QUEUE_abc_BEGIN\nM3W_QUEUE_abc_END:0\n'),'abc')
    assert value['queue_snapshot_verified'] and value['queue_lines']==[]
    assert not value['job_completion_inferred'] and value['jobs_submitted']==0


def test_failed_or_duplicate_markers_do_not_verify():
    for text,code in [('M3W_QUEUE_abc_BEGIN\nM3W_QUEUE_abc_END:1\n',1),
                      ('M3W_QUEUE_abc_BEGIN\nM3W_QUEUE_abc_BEGIN\nM3W_QUEUE_abc_END:0\n',0),
                      ('M3W_QUEUE_abc_END:0\nM3W_QUEUE_abc_BEGIN\n',0)]:
        assert not classify_response(dict(returncode=code,stdout=text),'abc')['queue_snapshot_verified']
