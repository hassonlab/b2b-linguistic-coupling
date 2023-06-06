from mne_bids import BIDSPath


class DerivativeBIDSPath(BIDSPath):
    def __init__(self, **kwargs):
        kwargs["check"] = False
        if "derivative" in kwargs:
            kwargs["root"] = kwargs["root"] + "/" + kwargs["derivative"]
            del kwargs["derivative"]
        super(DerivativeBIDSPath, self).__init__(**kwargs)

    def update(self, *, check=None, **kwargs):
        # elif isinstance(val, int):
        #     kwargs[key] = '{:02}'.format(val)

        for key, val in kwargs.items():
            setattr(self, f"_{key}", val)

        return self


def getparts(string: str) -> dict:
    parts = {}
    for part in string.split("_"):
        kv = part.split("-", 1)
        value = None
        if len(kv) > 1:
            value = kv[1]
            if value.isnumeric():
                value = int(value)
        parts[kv[0]] = value
    return parts


def aggparts(parts: dict) -> str:
    return "_".join(
        f"{key}-{value}" if value is not None else key for key, value in parts.items()
    )


def derivpath(
    filename: str,
    derivative: str,
    datatype=None,
    root="../dataset/derivatives",
    exclusions={"sub", "task", "datatype"},
) -> BIDSPath:
    """Get a derivative path from the filename."""
    base, extension = filename.rsplit(".", 1)
    parts = getparts(base)
    sub = parts["sub"]
    sub = "{:02d}".format(sub) if isinstance(sub, int) else sub
    suffix = "_".join(
        f"{key}-{value}" if value is not None else key
        for key, value in parts.items()
        if key not in exclusions
    )
    path = DerivativeBIDSPath(
        subject=sub,
        suffix=suffix,
        task=parts.get("task"),
        datatype=datatype,
        extension="." + extension,
        check=False,
        root=f"{root}/{derivative}/",
    )
    return path
