# Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


# This script is based on an original implementation by True Price.

import sqlite3
import sys

import cv2
import numpy as np

IS_PYTHON3 = sys.version_info[0] >= 3

MAX_IMAGE_ID = 2**31 - 1

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(MAX_IMAGE_ID)

CREATE_POSE_PRIORS_TABLE = """CREATE TABLE IF NOT EXISTS pose_priors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    position BLOB,
    coordinate_system INTEGER NOT NULL,
    position_covariance BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB,
    qvec BLOB,
    tvec BLOB)
"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = (
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"
)

CREATE_ALL = "; ".join(
    [
        CREATE_CAMERAS_TABLE,
        CREATE_IMAGES_TABLE,
        CREATE_POSE_PRIORS_TABLE,
        CREATE_KEYPOINTS_TABLE,
        CREATE_DESCRIPTORS_TABLE,
        CREATE_MATCHES_TABLE,
        CREATE_TWO_VIEW_GEOMETRIES_TABLE,
        CREATE_NAME_INDEX,
    ]
)

def load_cameras_params(calib_cameras_data_path):
    """Load camera parameters from a JSON file."""
    fs = cv2.FileStorage(str(calib_cameras_data_path), cv2.FILE_STORAGE_READ)
    num_cameras =  int(fs.getNode("nb_camera").real())
    camera_matrix_list = []
    distortion_coefficients_list = []

    w2c_list = []
    for cam_idx in range(num_cameras):
        cam_name = f"camera_{cam_idx}"
        camera_matrix = fs.getNode(cam_name).getNode("camera_matrix").mat()
        # with np.nditer(camera_matrix, op_flags=['readwrite']) as it:
        #     for x in it:
        #         if x != 0 and x !=1 :
        #             x *= 2 
        # camera_matrix_ = [it if (it == 1 or it == 0) else it/2  for it in camera_matrix ]
        camera_matrix_list.append(camera_matrix)
        distortion_coefficients = fs.getNode(cam_name).getNode("distortion_vector").mat()
        distortion_coefficients_list.append(distortion_coefficients)
        
        camera_pose_matrix = fs.getNode(cam_name).getNode("camera_pose_matrix").mat()
       

        # camera_pose_matrix[:3,3] *= scale/1000
        # camera_pose_matrix = c2w_0 @ camera_pose_matrix
        w2c = np.linalg.inv(camera_pose_matrix)
        # w2c[:3,3] *= 15.86/1000
        w2c_list.append(w2c.tolist())
    return camera_matrix_list, distortion_coefficients_list, w2c_list
def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) / MAX_IMAGE_ID
    return image_id1, image_id2


def array_to_blob(array):
    if IS_PYTHON3:
        return array.tobytes()
    else:
        return np.getbuffer(array)


def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)


class COLMAPDatabase(sqlite3.Connection):
    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = lambda: self.executescript(
            CREATE_CAMERAS_TABLE
        )
        self.create_descriptors_table = lambda: self.executescript(
            CREATE_DESCRIPTORS_TABLE
        )
        self.create_images_table = lambda: self.executescript(
            CREATE_IMAGES_TABLE
        )
        self.create_pose_priors_table = lambda: self.executescript(
            CREATE_POSE_PRIORS_TABLE
        )
        self.create_two_view_geometries_table = lambda: self.executescript(
            CREATE_TWO_VIEW_GEOMETRIES_TABLE
        )
        self.create_keypoints_table = lambda: self.executescript(
            CREATE_KEYPOINTS_TABLE
        )
        self.create_matches_table = lambda: self.executescript(
            CREATE_MATCHES_TABLE
        )
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def add_camera(
        self,
        model,
        width,
        height,
        params,
        prior_focal_length=False,
        camera_id=None,
    ):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (
                camera_id,
                model,
                width,
                height,
                array_to_blob(params),
                prior_focal_length,
            ),
        )
        return cursor.lastrowid

    def add_image(
        self,
        name,
        camera_id,
        image_id=None,
    ):
        cursor = self.execute(
            "INSERT INTO images VALUES (?, ?, ?)", (image_id, name, camera_id)
        )
        return cursor.lastrowid

    def add_pose_prior(
        self, image_id, position, coordinate_system=-1, position_covariance=None
    ):
        position = np.asarray(position, dtype=np.float64)
        if position_covariance is None:
            position_covariance = np.full((3, 3), np.nan, dtype=np.float64)
        self.execute(
            "INSERT INTO pose_priors VALUES (?, ?, ?, ?)",
            (
                image_id,
                array_to_blob(position),
                coordinate_system,
                array_to_blob(position_covariance),
            ),
        )

    def add_keypoints(self, image_id, keypoints):
        assert len(keypoints.shape) == 2
        assert keypoints.shape[1] in [2, 4, 6]

        keypoints = np.asarray(keypoints, np.float32)
        self.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
            (image_id,) + keypoints.shape + (array_to_blob(keypoints),),
        )

    def add_descriptors(self, image_id, descriptors):
        descriptors = np.ascontiguousarray(descriptors, np.uint8)
        self.execute(
            "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
            (image_id,) + descriptors.shape + (array_to_blob(descriptors),),
        )

    def add_matches(self, image_id1, image_id2, matches):
        assert len(matches.shape) == 2
        assert matches.shape[1] == 2

        if image_id1 > image_id2:
            matches = matches[:, ::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        self.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches),),
        )

    def add_two_view_geometry(
        self,
        image_id1,
        image_id2,
        matches,
        F=np.eye(3),
        E=np.eye(3),
        H=np.eye(3),
        qvec=np.array([1.0, 0.0, 0.0, 0.0]),
        tvec=np.zeros(3),
        config=2,
    ):
        assert len(matches.shape) == 2
        assert matches.shape[1] == 2

        if image_id1 > image_id2:
            matches = matches[:, ::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        F = np.asarray(F, dtype=np.float64)
        E = np.asarray(E, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        qvec = np.asarray(qvec, dtype=np.float64)
        tvec = np.asarray(tvec, dtype=np.float64)
        self.execute(
            "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (pair_id,)
            + matches.shape
            + (
                array_to_blob(matches),
                config,
                array_to_blob(F),
                array_to_blob(E),
                array_to_blob(H),
                array_to_blob(qvec),
                array_to_blob(tvec),
            ),
        )


def example_usage():
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", default="database.db")
    args = parser.parse_args()

    # if os.path.exists(args.database_path):
    #     print("ERROR: database path already exists -- will not modify it.")
    #     return

    # Open the database.

    db = COLMAPDatabase.connect(args.database_path)

    # For convenience, try creating all the tables upfront.

    db.create_tables()

    # Create dummy cameras.

    model1, width1, height1, params1 = (
        0,
        1024,
        768,
        np.array((1024.0, 512.0, 384.0)),
    )
    model2, width2, height2, params2 = (
        2,
        1024,
        768,
        np.array((1024.0, 512.0, 384.0, 0.1)),
    )

    camera_id1 = db.add_camera(model1, width1, height1, params1)
    camera_id2 = db.add_camera(model2, width2, height2, params2)

    # Create dummy images.

    image_id1 = db.add_image("image1.png", camera_id1)
    image_id2 = db.add_image("image2.png", camera_id1)
    image_id3 = db.add_image("image3.png", camera_id2)
    image_id4 = db.add_image("image4.png", camera_id2)

    # Create dummy keypoints.
    #
    # Note that COLMAP supports:
    #      - 2D keypoints: (x, y)
    #      - 4D keypoints: (x, y, theta, scale)
    #      - 6D affine keypoints: (x, y, a_11, a_12, a_21, a_22)

    num_keypoints = 1000
    keypoints1 = np.random.rand(num_keypoints, 2) * (width1, height1)
    keypoints2 = np.random.rand(num_keypoints, 2) * (width1, height1)
    keypoints3 = np.random.rand(num_keypoints, 2) * (width2, height2)
    keypoints4 = np.random.rand(num_keypoints, 2) * (width2, height2)

    db.add_keypoints(image_id1, keypoints1)
    db.add_keypoints(image_id2, keypoints2)
    db.add_keypoints(image_id3, keypoints3)
    db.add_keypoints(image_id4, keypoints4)

    # Create dummy matches.

    M = 50
    matches12 = np.random.randint(num_keypoints, size=(M, 2))
    matches23 = np.random.randint(num_keypoints, size=(M, 2))
    matches34 = np.random.randint(num_keypoints, size=(M, 2))

    db.add_matches(image_id1, image_id2, matches12)
    db.add_matches(image_id2, image_id3, matches23)
    db.add_matches(image_id3, image_id4, matches34)

    # Create dummy pose_priors.

    pos1 = np.random.rand(3, 1) * np.random.randint(10)
    pos2 = np.random.rand(3, 1) * np.random.randint(10)
    pos3 = np.random.rand(3, 1) * np.random.randint(10)

    cov3 = np.random.rand(3, 3) * np.random.randint(10)

    pose_prior1 = [image_id1, pos1, 1, None]
    pose_prior2 = [image_id2, pos2, -1, None]
    pose_prior3 = [image_id3, pos3, 0, cov3]

    db.add_pose_prior(*pose_prior1)
    db.add_pose_prior(*pose_prior2)
    db.add_pose_prior(*pose_prior3)

    # Convert unset covariance to nan matrix for later check
    pose_prior1[3] = np.full((3, 3), np.nan, dtype=np.float64)
    pose_prior2[3] = np.full((3, 3), np.nan, dtype=np.float64)

    # Commit the data to the file.

    db.commit()

    # Read and check cameras.

    rows = db.execute("SELECT * FROM cameras")

    camera_id, model, width, height, params, prior = next(rows)
    params = blob_to_array(params, np.float64)
    assert camera_id == camera_id1
    assert model == model1 and width == width1 and height == height1
    assert np.allclose(params, params1)

    camera_id, model, width, height, params, prior = next(rows)
    params = blob_to_array(params, np.float64)
    assert camera_id == camera_id2
    assert model == model2 and width == width2 and height == height2
    assert np.allclose(params, params2)

    # Read and check keypoints.

    keypoints = dict(
        (image_id, blob_to_array(data, np.float32, (-1, 2)))
        for image_id, data in db.execute("SELECT image_id, data FROM keypoints")
    )

    assert np.allclose(keypoints[image_id1], keypoints1)
    assert np.allclose(keypoints[image_id2], keypoints2)
    assert np.allclose(keypoints[image_id3], keypoints3)
    assert np.allclose(keypoints[image_id4], keypoints4)

    # Read and check matches.

    matches = dict(
        (pair_id_to_image_ids(pair_id), blob_to_array(data, np.uint32, (-1, 2)))
        for pair_id, data in db.execute("SELECT pair_id, data FROM matches")
    )

    assert np.all(matches[(image_id1, image_id2)] == matches12)
    assert np.all(matches[(image_id2, image_id3)] == matches23)
    assert np.all(matches[(image_id3, image_id4)] == matches34)

    # Read and check pose_priors

    rows = db.execute("SELECT * FROM pose_priors")

    img_id1, pos1, coord_sys1, cov1 = next(rows)
    img_id2, pos2, coord_sys2, cov2 = next(rows)
    img_id3, pos3, coord_sys3, cov3 = next(rows)

    assert pose_prior1[0] == img_id1
    assert pose_prior2[0] == img_id2
    assert pose_prior3[0] == img_id3

    assert pose_prior1[1].all() == blob_to_array(pos1, np.float64, (3, 1)).all()
    assert pose_prior2[1].all() == blob_to_array(pos2, np.float64, (3, 1)).all()
    assert pose_prior3[1].all() == blob_to_array(pos3, np.float64, (3, 1)).all()

    assert pose_prior1[2] == coord_sys1
    assert pose_prior2[2] == coord_sys2
    assert pose_prior3[2] == coord_sys3

    assert pose_prior1[3].all() == blob_to_array(cov1, np.float64, (3, 3)).all()
    assert pose_prior2[3].all() == blob_to_array(cov2, np.float64, (3, 3)).all()
    assert pose_prior3[3].all() == blob_to_array(cov3, np.float64, (3, 3)).all()

    # Clean up.

    db.close()

    if os.path.exists(args.database_path):
        os.remove(args.database_path)


if __name__ == "__main__":
    example_usage()
