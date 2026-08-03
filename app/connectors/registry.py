"""
app.connectors.registry
========================

Auto-discovery registry for connector plugins.

This module is what turns "drop a new file in app/connectors/plugins/"
into "the platform can now import from that merchant" with zero changes
to any other file. It scans the plugins package at startup, imports
every module in it, and registers every class found that subclasses
:class:`app.connectors.base.BaseConnector`.

Design notes
------------
* Discovery uses ``pkgutil.iter_modules`` + ``importlib.import_module``
  rather than requiring plugins to be manually listed anywhere — this
  is the actual mechanism behind the architecture's "no modification of
  the core application should be required" rule for new connectors.
* A single broken plugin module (syntax error, missing dependency,
  import-time exception) must not prevent every other connector from
  loading. Each module import is individually wrapped in
  exception handling; failures are logged and skipped, not raised,
  so one bad plugin can't take down the whole platform's ingestion
  capability.
* Registration is keyed by :attr:`BaseConnector.connector_key`, which
  must be unique — a duplicate key is treated as a configuration error
  and raised loudly at discovery time, since silently letting one
  connector shadow another would cause very confusing production
  behavior (imports silently going to the wrong source).
* The registry only stores *classes*, not instances — instantiation
  happens per-use via :meth:`ConnectorRegistry.create_connector`, since
  each connector instance needs a specific :class:`ConnectorPolicy`
  (which may differ per merchant even for the same connector class,
  e.g. two different Awin advertiser programs with different caching
  rules).
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from functools import lru_cache
from types import ModuleType

from app.connectors.base import BaseConnector, ConnectorPolicy

logger = logging.getLogger(__name__)

#: Fully-qualified package that is scanned for connector plugins.
#: Every ``.py`` module directly inside this package is imported and
#: inspected for BaseConnector subclasses.
PLUGINS_PACKAGE = "app.connectors.plugins"


class DuplicateConnectorKeyError(RuntimeError):
    """Raised when two connector classes declare the same
    ``connector_key``. This is always a configuration error requiring
    a developer fix, so it is raised rather than logged-and-skipped."""


class ConnectorNotFoundError(RuntimeError):
    """Raised when code asks the registry for a ``connector_key`` that
    was never discovered/registered."""


class ConnectorRegistry:
    """Holds the mapping of ``connector_key`` -> connector class,
    populated by :meth:`discover`.

    Application code should obtain the process-wide instance via
    :func:`get_registry` rather than constructing this directly, except
    in tests where an isolated registry (populated via
    :meth:`register` with hand-picked test connectors) is preferable
    to scanning the real plugins package.
    """

    def __init__(self) -> None:
        self._connectors: dict[str, type[BaseConnector]] = {}

    def register(self, connector_cls: type[BaseConnector]) -> None:
        """Manually register a connector class, bypassing discovery.

        Used both internally by :meth:`discover` and directly by tests
        that want to register a fake/mock connector without it living
        in the real plugins package.
        """
        key = connector_cls.connector_key
        if not key:
            logger.warning(
                "Skipping connector class %s: connector_key is empty.",
                connector_cls.__name__,
            )
            return

        existing = self._connectors.get(key)
        if existing is not None and existing is not connector_cls:
            raise DuplicateConnectorKeyError(
                f"connector_key '{key}' is already registered to "
                f"{existing.__module__}.{existing.__name__}; cannot also "
                f"register {connector_cls.__module__}.{connector_cls.__name__}."
            )

        self._connectors[key] = connector_cls
        logger.info(
            "Registered connector '%s' -> %s.%s",
            key,
            connector_cls.__module__,
            connector_cls.__name__,
        )

    def discover(self, package_name: str = PLUGINS_PACKAGE) -> int:
        """Import every module in ``package_name`` and register any
        :class:`BaseConnector` subclasses found.

        A module that fails to import (syntax error, missing
        third-party dependency, etc.) is logged and skipped rather than
        raised, so a single broken plugin cannot prevent the rest of
        the platform's connectors from loading.

        Returns
        -------
        int
            The number of connector classes newly registered.
        """
        try:
            package = importlib.import_module(package_name)
        except ImportError:
            logger.exception(
                "Could not import connector plugins package '%s'; "
                "no connectors were discovered.",
                package_name,
            )
            return 0

        registered_before = len(self._connectors)

        for module_info in pkgutil.iter_modules(package.__path__, prefix=f"{package_name}."):
            module_name = module_info.name
            try:
                module = importlib.import_module(module_name)
            except Exception:
                logger.exception(
                    "Failed to import connector plugin module '%s'; skipping it. "
                    "Other connectors will still be loaded.",
                    module_name,
                )
                continue

            self._register_connectors_in_module(module)

        registered_count = len(self._connectors) - registered_before
        logger.info(
            "Connector discovery complete: %d connector(s) registered from '%s'.",
            registered_count,
            package_name,
        )
        return registered_count

    def _register_connectors_in_module(self, module: ModuleType) -> None:
        """Inspect one already-imported module for BaseConnector
        subclasses defined in it, and register each one found."""
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, BaseConnector)
                and attr is not BaseConnector
                and attr.__module__ == module.__name__
            ):
                self.register(attr)

    def get_connector_class(self, connector_key: str) -> type[BaseConnector]:
        """Look up a registered connector class by key.

        Raises
        ------
        ConnectorNotFoundError
            If no connector with this key has been registered.
        """
        try:
            return self._connectors[connector_key]
        except KeyError as exc:
            raise ConnectorNotFoundError(
                f"No connector registered with connector_key='{connector_key}'. "
                f"Known keys: {sorted(self._connectors.keys())}"
            ) from exc

    def create_connector(
        self, connector_key: str, policy: ConnectorPolicy
    ) -> BaseConnector:
        """Instantiate a registered connector class with the given
        policy. This is the normal way the rest of the application
        (import pipeline, scheduler) should obtain a connector
        instance."""
        connector_cls = self.get_connector_class(connector_key)
        return connector_cls(policy=policy)

    def list_connector_keys(self) -> list[str]:
        """Return all currently registered connector keys, sorted, for
        display in the admin panel's connector list."""
        return sorted(self._connectors.keys())

    def __len__(self) -> int:
        return len(self._connectors)


@lru_cache(maxsize=1)
def get_registry() -> ConnectorRegistry:
    """Return the process-wide :class:`ConnectorRegistry` singleton,
    populated via :meth:`ConnectorRegistry.discover` on first access.

    Cached with ``lru_cache`` so plugin discovery (which does real
    filesystem/import work) happens exactly once per process.
    Application startup code (FastAPI's ``lifespan`` handler, the
    Celery worker's startup hook) should call this early so discovery
    failures surface immediately rather than on the first request that
    happens to need a connector.
    """
    registry = ConnectorRegistry()
    registry.discover()
    return registry

