"""NATS subject builders for inter-component routing.

Two routing channels connect the live pipeline:

  AlphaExecutor  --alpha_signal-->  PortfolioExecutor  --portfolio_signal-->  OMS

Both roots optionally carry a `namespace` token (e.g. a user id) so that, on a
shared NATS server, a consumer only receives its own tenant's traffic. NATS
wildcards match whole tokens, so the namespace must be its own token
(`alpha_signal.<ns>.<id>`) — it cannot be a within-token prefix.

These are separate from the dashboard-metric subjects MetricBuilder publishes
(those carry an `insert_prefix` namespace like `aegis_ts.*`).
"""

from __future__ import annotations

ALPHA_SIGNAL_ROOT = "alpha_signal"
PORTFOLIO_SIGNAL_ROOT = "portfolio_signal"
OMS_COMMAND_ROOT = "oms_command"


def alpha_signal_subject(alpha_id: str, namespace: str | None = None) -> str:
    """Subject an alpha publishes its signal on, for portfolio routing.

    With a `namespace` the id becomes its own token
    (`alpha_signal.<namespace>.<alpha_id>`) so a PortfolioExecutor can subscribe
    to only its own namespace; without one, the legacy flat subject is kept."""
    if namespace:
        return f"{ALPHA_SIGNAL_ROOT}.{namespace}.{alpha_id}"
    return f"{ALPHA_SIGNAL_ROOT}.{alpha_id}"


def alpha_signal_subscription(namespace: str | None = None) -> str:
    """Subscription a PortfolioExecutor uses to receive alpha signals. Scoped to
    `namespace` when set (`alpha_signal.<namespace>.>`), else `alpha_signal.*`."""
    if namespace:
        return f"{ALPHA_SIGNAL_ROOT}.{namespace}.>"
    return f"{ALPHA_SIGNAL_ROOT}.*"


def parse_alpha_id(subject: str, namespace: str | None = None) -> str:
    """Recover the alpha id from a signal subject (inverse of
    `alpha_signal_subject`). With a namespace the id is everything after the
    `alpha_signal.<namespace>.` prefix (rejoined so ids containing dots survive);
    otherwise it's the single token after `alpha_signal.`."""
    parts = subject.split(".")
    if namespace:
        return ".".join(parts[2:])
    return parts[1]


def portfolio_signal_subject(portfolio_id: str, namespace: str | None = None) -> str:
    """Subject a PortfolioExecutor publishes its aggregated target on, which the
    OMS subscribes to by exact portfolio id. Namespaced symmetrically with
    `alpha_signal_subject` so both ends scope to the same tenant token."""
    if namespace:
        return f"{PORTFOLIO_SIGNAL_ROOT}.{namespace}.{portfolio_id}"
    return f"{PORTFOLIO_SIGNAL_ROOT}.{portfolio_id}"


def oms_command_subject(oms_id: str, namespace: str | None = None) -> str:
    """Control-plane subject an operator publishes commands on for a specific
    OMS instance (e.g. rebalance).

    This is a third purpose distinct from the routing roots above (flow =
    data) and the MetricBuilder dashboard subjects (aegis = metrics): here we
    deliver control. Namespaced symmetrically so a shared broker scopes per
    tenant."""
    if namespace:
        return f"{OMS_COMMAND_ROOT}.{namespace}.{oms_id}"
    return f"{OMS_COMMAND_ROOT}.{oms_id}"
