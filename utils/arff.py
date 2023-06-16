from river.stream import iter_arff
from scipy.io.arff import arffread
from river import base
from river.stream import utils as ru
from river.datasets import base as dsbase


class ArffStream(base.Dataset):
    def __init__(self, filepath, compression="infer"):
        buffer = filepath
        if not hasattr(buffer, "read"):
            buffer = ru.open_filepath(buffer, compression)

        try:
            rel, attrs = arffread.read_header(buffer)
        except ValueError as e:
            msg = f"Error while parsing header, error was: {e}"
            raise arffread.ParseArffError(msg)

        features = [attr.name for attr in attrs[:-1]]
        feat_types = [
            float if isinstance(attr, arffread.NumericAttribute) else None
            for attr in attrs
        ]

        target = [attr.name for attr in attrs[-1]]
        target_range = [attr.range for attr in attrs[-1]]
        feat_types = [
            float if isinstance(attr, arffread.NumericAttribute) else None
            for attr in attrs
        ]

    super().__init__(
        n_features=30,
        task=base.BINARY_CLF,
        url="https://maxhalford.github.io/files/datasets/creditcardfraud.zip",
        size=150_828_752,
        filename="creditcard.csv",
    )


def iter_arff(
    filepath_or_buffer, target: str = None, compression="infer"
) -> base.typing.Stream:
    """Iterates over rows from an ARFF file.

    Parameters
    ----------
    filepath_or_buffer
        Either a string indicating the location of a file, or a buffer object that has a
        `read` method.
    target
        Name of the target field.
    compression
        For on-the-fly decompression of on-disk data. If this is set to 'infer' and
        `filepath_or_buffer` is a path, then the decompression method is inferred for the
        following extensions: '.gz', '.zip'.

    """

    # If a file is not opened, then we open it
    buffer = filepath_or_buffer
    if not hasattr(buffer, "read"):
        buffer = ru.open_filepath(buffer, compression)

    try:
        rel, attrs = arffread.read_header(buffer)
    except ValueError as e:
        msg = f"Error while parsing header, error was: {e}"
        raise arffread.ParseArffError(msg)

    print(attrs)
    names = [attr.name for attr in attrs]
    range = [attr.range for attr in attrs]
    print(range)
    types = [
        float if isinstance(attr, arffread.NumericAttribute) else None for attr in attrs
    ]

    for r in buffer:
        if len(r) == 0:
            continue

        line = r.rstrip().split(",")
        x = {
            name: typ(val) if typ else val
            for name, typ, val in zip(names, types, r.rstrip().split(","))
        }
        try:
            y = x.pop(target) if target else None
        except KeyError as e:
            print(r)
            raise e

        yield x, y

    # Close the file if we opened it
    if buffer is not filepath_or_buffer:
        buffer.close()


if __name__ == "__main__":
    gen = iter_arff("../data.arff")
    for x, y in gen:
        print(x)
