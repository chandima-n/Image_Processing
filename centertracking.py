from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentTracker:
    def __init__(self, max_disapp=50, max_dist=50):
        # Initialize the next unique object ID along with two ordered dictionaries
        # for tracking detected objects until they disappear.
        self.newvehi_id = 1
        self.vehicle = OrderedDict()
        self.disappear = OrderedDict()
        self.max_disapp = max_disapp
        self.max_dist = max_dist

    def appear_vehi(self, centroid):
        # When a new vehicle appears, assign it the next available ID.
        self.vehicle[self.newvehi_id] = centroid
        self.disappear[self.newvehi_id] = 1
        self.newvehi_id += 1

    def disappear_vehi(self, vehi_id):
        # Remove the vehicle from tracking when it has vanished.
        del self.vehicle[vehi_id]
        del self.disappear[vehi_id]

    def update(self, rect):
        # If no bounding boxes are detected, increase the disappear count for each vehicle.
        if len(rect) == 0:
            for vehi_id in list(self.disappear.keys()):
                self.disappear[vehi_id] += 1
                # If a vehicle has exceeded the maximum allowed disappearances, remove it.
                if self.disappear[vehi_id] > self.max_disapp:
                    self.disappear_vehi(vehi_id)
            return self.vehicle

        # Initialize an array of new centroids for the current frame.
        new_centroids = np.zeros((len(rect), 2), dtype="int")

        # Find the centers of bounding boxes.
        for (i, (box_x, box_y, box_w, box_h)) in enumerate(rect):
            cen_x = int((box_x + box_w) / 2.0)
            cen_y = int((box_y + box_h) / 2.0)
            new_centroids[i] = (cen_x, cen_y)

        # If there are no existing vehicles, register all new objects as new vehicles.
        if len(self.vehicle) == 0:
            for i in range(len(new_centroids)):
                self.appear_vehi(new_centroids[i])

        else:
            # Collect the set of object IDs and their current centroids.
            vehi_ids = list(self.vehicle.keys())
            exist_centroids = list(self.vehicle.values())

            # Compute the distance between each pair of existing and new centroids.
            distance = dist.cdist(np.array(exist_centroids), new_centroids)

            # Sort distances to find the best matches.
            rows = distance.min(axis=1).argsort()
            columns = distance.argmin(axis=1)[rows]

            # Track which rows and columns have been checked.
            used_rows = set()
            used_columns = set()

            # Assign new centroids to existing vehicle IDs.
            for (row, column) in zip(rows, columns):
                if row in used_rows or column in used_columns:
                    continue
                if distance[row, column] > self.max_dist:
                    continue

                vehi_id = vehi_ids[row]
                self.vehicle[vehi_id] = new_centroids[column]
                self.disappear[vehi_id] = 0
                used_rows.add(row)
                used_columns.add(column)

            # Determine which existing objects were not matched.
            unused_rows = set(range(distance.shape[0])).difference(used_rows)
            unused_columns = set(range(distance.shape[1])).difference(used_columns)

            # For each unmatched existing object, increase its disappear count.
            if distance.shape[0] >= distance.shape[1]:
                for row in unused_rows:
                    vehi_id = vehi_ids[row]
                    self.disappear[vehi_id] += 1
                    if self.disappear[vehi_id] > self.max_disapp:
                        self.disappear_vehi(vehi_id)
            else:
                # Register new centroids as new vehicles.
                for column in unused_columns:
                    self.appear_vehi(new_centroids[column])

        return self.vehicle
