from belay.demo import main


def test_demo_reads_its_config_and_prints_every_policy(capsys):
    out = main([])
    assert set(out) == {"none", "every", "periodic", "indep", "belay", "belay+replan"}
    assert "BELAY, re-planning after flags" in capsys.readouterr().out
