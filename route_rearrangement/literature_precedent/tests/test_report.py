"""The audit labelling must work on *real* FrequenTree templates, not just idealised ones.

Both bugs this file guards against were silent: a label that never fires reads as "the corpus
does not contain that chemistry" rather than "the predicate is broken".  The templates below
are verbatim from the PaRoutes centers cache, in FrequenTree's own dialect — charge written
``[N;+1;...]``, fragments wrapped in grouping parens — which is precisely what SMARTS-based
matching failed on.
"""

import pytest

from route_rearrangement.literature_precedent import report


# verbatim from ~/Downloads/paroutes_all/centers (tree all-0), FrequenTree dialect
REAL_NITRATION = ("([N;+1;D3;H0:1](-[O;-1;D1;H0:2])(-[O;D1;H1;+0:3])=[O;D1;H0;+0])."
                  "([c;H1;D2;+0:4])>>"
                  "([N;+1;D3;H0:1](-[O;-1;D1;H0:2])(=[O;D1;H0;+0:3])-[c;D3;H0;+0:4])")
REAL_NITRO_REDUCTION = ("([N;+1;D3;H0:1](-[O;-1;D1;H0:2])(=[O;D1;H0;+0:3])-[c;D3;H0;+0:4])>>"
                        "([N;D3;H2;+0:1]-[c;D3;H0;+0:4])")


def test_labels_fire_on_real_frequentree_dialect():
    assert report.label_key(REAL_NITRO_REDUCTION) == ["nitro_reduction"]


def test_nitration_is_not_labelled_a_reduction():
    """Direction matters: the same atoms in the other order must not earn the label."""
    assert "nitro_reduction" not in report.label_key(REAL_NITRATION)


# A genuine Boc removal: the carbamate carbon is in the reaction centre, so it is mapped.
BOC_REMOVAL = ("([N;D3;H0;+0:1]-[C;D3;H0;+0:2](=[O;D1;H0;+0:3])-[O;H0;D2;+0:4]"
               "-[C;D4;H0;+0:5](-[C;H3;D1;+0])(-[C;H3;D1;+0])-[C;H3;D1;+0])>>"
               "([N;D3;H1;+0:1])")
BOC_INSTALL = ("([N;D3;H1;+0:1])>>"
               "([N;D3;H0;+0:1]-[C;D3;H0;+0:2](=[O;D1;H0;+0:3])-[O;H0;D2;+0:4]"
               "-[C;D4;H0;+0:5](-[C;H3;D1;+0])(-[C;H3;D1;+0])-[C;H3;D1;+0])")
# Verbatim from the corpus: a sulfonylation whose substrate merely *carries* an untouched Boc.
# The Boc appears on the reactant side and not the product side — which is true of every
# spectator group — so a naive "present left, absent right" rule calls it a deprotection.
SULFONYLATION_ON_BOC_SUBSTRATE = (
    "([Cl;D1;H0;+0]-[S;D4;H0;+0:1](=[O;D1;H0;+0:2])=[O;D1;H0;+0:3])."
    "([N;D3;H0;+0:4]-[C;D3;H0;+0](=[O;D1;H0;+0])-[O;H0;D2;+0]"
    "-[C;D4;H0;+0](-[C;H3;D1;+0])(-[C;H3;D1;+0])-[C;H3;D1;+0])>>"
    "([S;D4;H0;+0:1](-[N;D3;H0;+0:4])(=[O;D1;H0;+0:2])=[O;D1;H0;+0:3])")


@pytest.mark.parametrize("template,expected", [
    ("([c:1]Cl).([NH2:2]C)>>([c:1][NH:2]C)", "snar_like"),
    ("([c:1]Br).([c:2]B(O)O)>>([c:1][c:2])", "cross_coupling"),
    (BOC_REMOVAL, "deprotection_carbamate"),
    (BOC_INSTALL, "protection_carbamate"),
])
def test_each_audit_label(template, expected):
    assert expected in report.label_key(template)


def test_spectator_protecting_group_is_not_a_deprotection():
    """The regression that mattered: 21 significant pairs were mislabelled this way."""
    assert "deprotection_carbamate" not in report.label_key(SULFONYLATION_ON_BOC_SUBSTRATE)


def test_protection_and_deprotection_are_never_both_assigned():
    for t in (BOC_REMOVAL, BOC_INSTALL):
        labels = report.label_key(t)
        assert not {"protection_carbamate", "deprotection_carbamate"} <= set(labels)


def test_non_template_keys_are_ignored():
    """Coarse synthon keys are atom-token bags, not SMARTS — labelling must decline, not guess."""
    assert report.label_key("C[];N[];O[]>>C[];N[];O[]") == []
    assert report.label_key("not a template") == []


def test_every_auditable_motif_names_known_labels():
    """A typo in AUDITABLE_MOTIFS would silently make a motif unrecoverable forever."""
    producible = set()
    for t in (REAL_NITRO_REDUCTION, "([c:1]Cl).([NH2:2]C)>>([c:1][NH:2]C)",
              "([c:1]Br).([c:2]B(O)O)>>([c:1][c:2])", BOC_REMOVAL, BOC_INSTALL):
        producible.update(report.label_key(t))
    for motif, (first, second) in report.AUDITABLE_MOTIFS.items():
        assert first in producible, f"{motif}: no template produces label {first!r}"
        assert second in producible, f"{motif}: no template produces label {second!r}"


def test_audited_motifs_exist_in_the_catalogue():
    from route_rearrangement.motifs import MOTIFS
    names = {m.name for m in MOTIFS}
    assert set(report.AUDITABLE_MOTIFS) <= names
