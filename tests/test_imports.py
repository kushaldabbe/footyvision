def test_imports():
    try:
        import src.footyvision
        assert True
    except ImportError:
        assert False
