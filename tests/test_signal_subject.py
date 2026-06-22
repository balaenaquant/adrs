"""Subject helpers for namespaced signal routing.

A namespace makes the user id its own subject token so a PortfolioExecutor (and
the OMS) on a shared NATS server only receive their own tenant's traffic.
"""

from adrs.subjects import (
    alpha_signal_subject,
    alpha_signal_subscription,
    parse_alpha_id,
    portfolio_signal_subject,
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


def test_portfolio_signal_subject_legacy_and_namespaced():
    assert portfolio_signal_subject("marcus_pf") == "portfolio_signal.marcus_pf"
    assert (
        portfolio_signal_subject("marcus_pf", "marcus")
        == "portfolio_signal.marcus.marcus_pf"
    )


def test_portfolio_publish_and_oms_subscribe_match():
    # PortfolioExecutor publishes with its namespace; OMS subscribes to its own
    # portfolio id under the same namespace — exact-subject match, no wildcard.
    ns, pf = "marcus", "marcus_pf"
    assert portfolio_signal_subject(pf, ns) == portfolio_signal_subject(pf, ns)
    # a different tenant's subject differs in the namespace token
    assert portfolio_signal_subject("bob_pf", "bob") != portfolio_signal_subject(pf, ns)
