"""Subject helpers for namespaced alpha-signal routing.

A namespace makes the user id its own subject token so a PortfolioExecutor on a
shared NATS server only receives its own tenant's signals.
"""

from adrs.execution.executor import (
    alpha_signal_subject,
    alpha_signal_subscription,
    parse_alpha_id,
)


def test_legacy_subject_without_namespace():
    assert alpha_signal_subject("marcus_myalpha") == "alpha_signal.marcus_myalpha"
    assert alpha_signal_subscription() == "alpha_signal.*"


def test_namespaced_subject_makes_user_a_token():
    assert (
        alpha_signal_subject("marcus_myalpha", "marcus")
        == "alpha_signal.marcus.marcus_myalpha"
    )
    # `>` (not `*`) so the alpha id may itself span tokens
    assert alpha_signal_subscription("marcus") == "alpha_signal.marcus.>"


def test_subscription_does_not_match_other_namespace():
    # sanity on the token shape the wildcard relies on: bob's subject has a
    # different second token, so `alpha_signal.marcus.>` cannot match it.
    bob = alpha_signal_subject("bob_alpha", "bob")
    assert not bob.startswith("alpha_signal.marcus.")


def test_parse_alpha_id_roundtrip():
    for ns in (None, "marcus"):
        subject = alpha_signal_subject("marcus_myalpha", ns)
        assert parse_alpha_id(subject, ns) == "marcus_myalpha"


def test_parse_alpha_id_preserves_dotted_id():
    subject = alpha_signal_subject("team.alpha.v2", "marcus")
    assert parse_alpha_id(subject, "marcus") == "team.alpha.v2"
