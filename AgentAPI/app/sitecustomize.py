import os

_OVERRIDE_KEYS = (
    "REDIS_URI_OVERRIDE",
    "EXTERNAL_REDIS_URI",
    "HOST_REDIS_URI",
    "REDIS_URI",
)

for _key in _OVERRIDE_KEYS:
    _value = os.getenv(_key)
    if _value:
        os.environ["REDIS_URI"] = _value
        break
