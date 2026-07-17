def test_synthesis_extraction_importable():
    from route_rearrangement import deps  # noqa: F401
    from synthesis_extraction.dependency.schedule import ScheduleLattice
    from synthesis_extraction.dependency.route_graph import build_route_graph  # noqa: F401

    lat = ScheduleLattice([1, 2, 3], [(3, 1), (2, 1)])
    assert lat.count() == 2  # 3 and 2 free relative to each other, both before 1


def test_rdchiral_importable():
    from rdchiral.template_extractor import extract_from_reaction  # noqa: F401
    from rdchiral.main import rdchiralRun  # noqa: F401
