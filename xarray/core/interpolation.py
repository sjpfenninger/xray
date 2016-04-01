import collections
import itertools

import numpy as np

from .dataarray import DataArray


def map_coordinates(source, target, keep_attrs=False, **kwargs):
    """
    Uses ``scipy.ndimage.interpolation.map_coordinates`` to map the
    source data array to the given target coordinates using spline
    interpolation. Target must consist of numerical coordinates only.

    Parameters
    ------------
    source : DataArray
        Gridded source data.
    target : dict
        Target coordinates; mapping from dimension names to coordinate
        values to sample at.
    keep_attrs : boolean, optional
        If True, attributes are copied from source (default is False).
    **kwargs
        Additional keyword args are passed to
        ``scipy.ndimage.interpolation.map_coordinates``.

    Returns
    ---------
    interpolated : DataArray
        Data array with interpolated values. Coordinates and dimensions
        are copied from source.

    """
    from scipy.interpolate import interp1d
    from scipy import ndimage

    # Set up the interpolators to map coordinates onto array indices
    interpolators = {}
    for dim_name in target.keys():
        dim = source.coords[dim_name]
        if not np.issubdtype(dim.dtype, np.number):
            raise ValueError('Only numerical dimensions '
                             'can be interpolated on.')
        try:
            interpolators[dim_name] = interp1d(dim, list(range(len(dim))))
        except ValueError:  # Raised if there is only one entry
            # 0 is the only index that exists
            interpolators[dim_name] = lambda x: 0

    # Set up the array indices on which to interpolate,
    # and the final coordinates to apply to the result
    indices = collections.OrderedDict()
    coords = collections.OrderedDict()
    for d in source.dims:
        if d not in target.keys():
            # Choose all entries from non-interpolated dimensions
            indices[d] = list(range(len(source.coords[d])))
            coords[d] = source.coords[d]
        else:
            # Choose selected entries from interpolated dimensions
            indices[d] = [interpolators[d](i) for i in target[d]]
            coords[d] = target[d]

    # Generate array of all coordinates
    # Shape should be n(dims) * n(product of all interpolators)
    coordinates = np.array(list(zip(
        *itertools.product(*[i for i in indices.values()]))
    ))

    interp_array = ndimage.map_coordinates(source.values, coordinates,
                                           **kwargs)

    # Reshape the resulting array according to the target coordinates
    result_shape = [len(i) for i in indices.values()]
    attrs = source.attrs if keep_attrs else None  # Copy attrs if asked for
    return DataArray(interp_array.reshape(result_shape),
                     coords=coords, attrs=attrs)
