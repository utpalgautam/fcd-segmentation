"""Skull-stripping interface; external backend intentionally not bundled."""


def skull_strip(*args, **kwargs):
    raise NotImplementedError(
        "Configure and validate a supported skull-stripping backend first."
    )
