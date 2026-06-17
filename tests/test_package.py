def test_package_exposes_version():
    import llm_router

    assert isinstance(llm_router.__version__, str)
    assert llm_router.__version__
