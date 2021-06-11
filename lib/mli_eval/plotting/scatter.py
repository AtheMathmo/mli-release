import matplotlib.pyplot as plt


def mscatter(x,y,ax=None, m=None, **kw):
    """Pyplot scatter with multiple marker styles in each plot call
    """
    import matplotlib.markers as mmarkers
    if not ax: ax=plt.gca()
    sc = ax.scatter(x,y,**kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    else:
        raise ValueError("Invalid markers of length {} for data of length {}".format(len(m), len(x)))
    return sc
