class trackingvehicle:
   def __init__(self, vehi_ID, centroid):
      # keep the object ID and next initialize the list of centers
      # using the existing centrs
      self.vehi_ID = vehi_ID
      self.centroids = [centroid]

      
      self.counted = False

