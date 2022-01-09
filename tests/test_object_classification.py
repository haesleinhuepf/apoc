import numpy as np
def test_object_classification():
    image      = np.asarray([[0,0,1,1,2,2,3,3,3,3,3]])
    labels     = np.asarray([[0,0,1,1,1,1,2,2,2,2,2]])
    annotation = np.asarray([[0,0,0,0,2,0,0,0,0,1,1]])
    
    feature_definition = """
            area
            min_intensity max_intensity sum_intensity mean_intensity standard_deviation_intensity
            mass_center_x mass_center_y mass_center_z
            centroid_x centroid_y centroid_z
            max_distance_to_centroid max_distance_to_mass_center
            mean_max_distance_to_centroid_ratio mean_max_distance_to_mass_center_ratio
            touching_neighbor_count average_distance_of_touching_neighbors average_distance_of_n_nearest_neighbors
            """.replace("\n", " ")
    
    import apoc
    oc = apoc.ObjectClassifier()
    oc.train(feature_definition, labels, annotation, image)
    oc.predict(labels, image)
