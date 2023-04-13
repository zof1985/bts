# BTS BIOENGINEERING IMPORTING MODULE


#! IMPORTS


import os
import struct
from io import BufferedReader
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["read_tdf"]


#! METHODS


def _read_tracks(
    fid: BufferedReader,
    n_frames: int,
    n_tracks: int,
    by_frame: bool,
    size: int,
    has_labels: bool = True,
):
    """
    internal method used to extract 3D tracks from tdf file.

    Parameters
    ----------
    fid: BufferedReader
        the buffer containing the data to be extracted.

    nFrames: int
        the number of samples denoting the tracks.

    nTracks: int
        the number of tracks defining the output.

    by_frame: bool
        should the data be read by frame or by track?

    size: int
        the expected number of channels for each track

    has_labels: bool
        used on untracked force platforms to avoid reading labels.

    Returns
    -------
    tracks: numpy.ndarray
        a 2D array with the extracted tracks

    labels: numpy.ndarray
        a 1D array with the labels of each tracks column

    index: numpy.ndarray
        the time index of each track row
    """

    # prepare the arrays for the tracks and the labels
    labels: list[str] = [""] * n_tracks
    tracks = np.ones((n_frames, size * n_tracks)) * np.nan

    # read the data
    for trk in range(n_tracks):
        # get the label
        if has_labels:
            lbls = struct.unpack("256B", fid.read(256))
            lbls = tuple(chr(i) for i in lbls)
            labels[trk] = "".join(lbls).split(chr(0), 1)[0]

        # read data
        if by_frame:
            n = size * n_tracks * n_frames
            segments = struct.unpack("%if" % n, fid.read(n * 4))
            tracks = np.array(segments)
            tracks = tracks.reshape(n_frames, size * n_tracks).T

        # read by track
        else:
            n = struct.unpack("i", fid.read(4))[0]
            fid.seek(4, 1)
            segments = struct.unpack(f"{2 * n}i", fid.read(8 * n))
            segments = np.array(segments).reshape(n, 2).T
            cols = np.atleast_2d(np.arange(size) + size * trk)
            for s in np.arange(n):
                for r in np.arange(segments[1, s]) + segments[0, s]:
                    vals = fid.read(4 * size)
                    if r < tracks.shape[0]:
                        tracks[r, cols] = struct.unpack(f"{size}f", vals)

    # return the tracks
    return tracks, labels


def _get_point3d(
    fid: BufferedReader,
    info: dict[str, Any],
):
    """
    read Point3D tracks data from the provided tdf file.

    Paramters
    ---------
    fid: BufferedReader
        the file buffer from which the data have to be extracted

    info: dict
        a dict extracted from the tdf file reading with the info
        required to extract Point3D data from it.

    Returns
    -------
    out: dict
        a dict of Marker3D and Link3D objects.
    """

    # get the file read
    fid.seek(info["Offset"], 0)
    frames, freq, time, n_tracks = struct.unpack("iifi", fid.read(16))

    # calibration data (read but not exported)
    A = np.array(struct.unpack("3f", fid.read(12)))
    B = np.array(struct.unpack("9f", fid.read(36))).reshape(3, 3).T
    C = np.array(struct.unpack("3f", fid.read(12)))
    fid.seek(4, 1)

    # check if links exists
    if info["Format"] in [1, 3]:
        (n_links,) = struct.unpack("i", fid.read(4))
        fid.seek(4, 1)
        links = struct.unpack("%ii" % (2 * n_links), fid.read(8 * n_links))
        links = np.reshape(links, (len(links) // 2, 2))
    else:
        links = []

    # check if the file has to be read by frame or by track
    by_frame = info["Format"] in [3, 4]
    by_track = info["Format"] in [1, 2]
    if not by_frame and not by_track:
        raise IOError(f"Invalid 'Format' info {info['Format']}")

    # read the data
    tracks, labels = _read_tracks(fid, frames, n_tracks, by_frame, 3)

    # generate the links
    links = [[labels[i], labels[j]] for i, j in links]

    # generate the output markers
    points: dict[str, pd.DataFrame] = {}
    cols = pd.MultiIndex.from_tuples([(i, "m") for i in ["X", "Y", "Z"]])
    idx = (np.arange(tracks.shape[0]) / freq).astype(float) + time
    idx = pd.Index(idx, name="Time")
    for i, lbl in enumerate(labels):
        data = tracks[:, np.arange(3) + i * 3]
        points[lbl] = pd.DataFrame(data=data, index=idx, columns=cols)

    return points, links


def _get_force3d(
    fid: BufferedReader,
    info: dict[str, Any],
):
    """
    read Force3D tracks data from the provided tdf file.

    Paramters
    ---------
    fid: BufferedReader
        the file buffer from which the data have to be extracted

    info: dict
        a dict extracted from the tdf file reading with the info
        required to extract Point3D data from it.

    Returns
    -------
    fp: dict
        a dict of sensors.ForcePlatform3D objects.
    """
    # get the file read
    fid.seek(info["Offset"], 0)
    n_tracks, freq, time, frames = struct.unpack("iifi", fid.read(16))

    # calibration data (read but not exported)
    A = np.array(struct.unpack("3f", fid.read(12)))
    B = np.array(struct.unpack("9f", fid.read(36))).reshape(3, 3).T
    C = np.array(struct.unpack("3f", fid.read(12)))
    fid.seek(4, 1)

    # check if the file has to be read by frame or by track
    by_frame = info["Format"] in [2]
    by_track = info["Format"] in [1]
    if not by_frame and not by_track:
        raise IOError("Invalid 'Format' info {info['Format']}")

    # read the data
    trks, lbls = _read_tracks(fid, frames, n_tracks, by_frame, 9)

    # generate the output dict
    src = ["Point"] * 3 + ["Force"] * 3 + ["Moment"] * 3
    dim = ["X", "Y", "Z"] * 3
    unt = ["m"] * 3 + ["N"] * 3 + ["N*m"] * 3
    cols = [(s, d, u) for s, d, u in zip(src, dim, unt)]
    cols = pd.MultiIndex.from_tuples(cols)
    idx = (np.arange(trks.shape[0]) / freq).astype(float) + time
    idx = pd.Index(idx, name="Time")
    platforms = {}
    for i, lbl in enumerate(lbls):
        platforms[lbl] = pd.DataFrame(
            data=trks[:, np.arange(9) + 9 * i],
            index=idx,
            columns=cols,
        )

    return platforms


def _get_emg(
    fid: BufferedReader,
    info: dict[str, Any],
):
    """
    read EMG tracks data from the provided tdf file.

    Paramters
    ---------
    fid:BufferedReader
        the file buffer containing the data to be extracted

    info: dict
        a dict extracted from the tdf file reading with the info
        required to extract Point3D data from it.

    Returns
    -------
    channels: dict
        a dict with all the EMG channels provided as simbiopy.EmgSensor.
    """
    # get the file read
    fid.seek(info["Offset"], 0)
    trks, freq, time, frames = struct.unpack("iifi", fid.read(16))

    # check if the file has to be read by frame or by track
    by_frame = info["Format"] in [2]
    by_track = info["Format"] in [1]
    if not by_frame and not by_track:
        raise IOError(f"Invalid 'Format' info {info['Format']}")

    # read the data
    fid.read(trks * 2)
    trks, lbls = _read_tracks(fid, frames, trks, by_frame, 1)

    # generate the output dict
    cols = pd.MultiIndex.from_tuples([(l, "V") for l in lbls])
    idx = (np.arange(trks.shape[0]) / freq).astype(float) + time
    idx = pd.Index(idx, name="Time")
    return {'EMG': pd.DataFrame(data=trks, index=idx, columns=cols)}


def _get_imu(
    fid: BufferedReader,
    info: dict[str, Any],
):
    """
    read IMU tracks data from the provided tdf file.

    Paramters
    ---------
    fid:BufferedReader
        the file buffer containing the data to be extracted

    info: dict
        a dict extracted from the tdf file reading with the info
        required to extract Point3D data from it.

    Returns
    -------
    points: dict
        a dict with all the tracks provided as simbiopy.Imu3D objects.
    """
    # check if the file has to be read by frame or by track
    if not info["Format"] in [5]:
        raise IOError("Invalid 'Format' info {}".format(info["Format"]))

    # get the file read
    fid.seek(info["Offset"], 0)
    tracks, frames, freq, time = struct.unpack("iifi", fid.read(16))

    # read the data
    fid.seek(2, 1)
    trks, lbls = _read_tracks(fid, frames, tracks, False, 9)

    # generate the output dict
    src = ["Accelerometer"] * 3 + ["Gyroscope"] * 3 + ["Magnetometer"] * 3
    dim = ["X", "Y", "Z"] * 3
    unt = ["m/s^2"] * 3 + ["deg/s"] * 3 + ["nT"] * 3
    cols = [(s, d, u) for s, d, u in zip(src, dim, unt)]
    cols = pd.MultiIndex.from_tuples(cols)
    idx = (np.arange(trks.shape[0]) / freq).astype(float) + time
    idx = pd.Index(idx, name="Time")
    imus = {}
    for i, lbl in enumerate(lbls):
        imus[lbl] = pd.DataFrame(
            data=trks[:, np.arange(9) + 9 * i],
            index=idx,
            columns=cols,
        )

    return imus


def read_tdf(
    path: str,
):
    """
    Return the readings from a .tdf file as dicts of 3D objects.

    Parameters
    ----------
    path: str
        an existing tdf path.

    Returns
    -------
    a dict containing the distinct data properly arranged by type.
    """

    # private signature
    tdf_signature = "41604B82CA8411D3ACB60060080C6816"

    # check the validity of the entered path
    assert os.path.exists(path), path + " does not exist."
    assert path[-4:] == ".tdf", path + ' must be an ".tdf" path.'

    # check the available data
    points = {}
    links = np.atleast_2d([])
    forceplatforms = {}
    emgchannels = {}
    imus = {}
    fid = open(path, "rb")
    try:
        # check the signature
        sig = struct.unpack("IIII", fid.read(16))
        sig = "".join(["{:08x}".format(b) for b in sig])
        if sig != tdf_signature.lower():
            raise IOError("invalid file")

        # get the number of entries
        _, n_entries = struct.unpack("Ii", fid.read(8))
        assert n_entries > 0, "The file specified contains no data."

        # reference indices
        ids = {
            "Point3D": 5,
            "ForcePlatform3D": 12,
            "EmgChannel": 11,
            "IMU": 17,
        }

        # check each entry to find the available blocks
        next_entry_offset = 40
        blocks = []
        for _ in range(n_entries):
            if -1 == fid.seek(next_entry_offset, 1):
                raise IOError("Error: the file specified is corrupted.")

            # get the data types
            block_info = struct.unpack("IIii", fid.read(16))
            block_labels = ["Type", "Format", "Offset", "Size"]
            bi = dict(zip(block_labels, block_info))

            # retain only valid block types
            if bi["Type"] in list(ids.values()):
                blocks += [bi]

            # update the offset
            next_entry_offset = 272

        # read the available data
        for b in blocks:
            if b["Type"] == ids["Point3D"]:
                points, links = _get_point3d(fid, b)
            elif b["Type"] == ids["ForcePlatform3D"]:
                forceplatforms = _get_force3d(fid, b)
            elif b["Type"] == ids["EmgChannel"]:
                emgchannels = _get_emg(fid, b)
            elif b["Type"] == ids["EmgChannel"]:
                imus = _get_imu(fid, b)

    finally:
        fid.close()

    return {
        "Point3D": points,
        "Link": links,
        "ForcePlatform3D": forceplatforms,
        "EmgChannel": emgchannels,
        "IMU": imus,
    }
