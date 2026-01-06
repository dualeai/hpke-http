# Vulture whitelist for false positives
# These variables are required by protocols/interfaces but unused by design

# Async context manager protocol requires these args even if unused
exc_type  # noqa
exc_val  # noqa
exc_tb  # noqa
