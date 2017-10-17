import glob
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split  # for sklearn > 0.17 use sklearn.model_selection instead
from helper_functions import *
import time
from styx_msgs.msg import TrafficLight
import os
import pickle
import rospy

class TLClassifier(object):
    def __init__(self):
        # Set paramters for feature extraction
        self.color_space = 'RGB'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb ...
        self.orient = 12  # HOG orientations, usually between 6 and 12
        self.pix_per_cell = 8  # 14  # HOG pixels per cell
        self.cell_per_block = 2  # HOG cells per block
        self.hog_channel = 1  # 1  # Can be 0, 1, 2, or "ALL"
        self.spatial_size = (8, 8)  # (32, 32)  # Spatial binning dimensions
        self.hist_bins = 64  # 196  # Number of histogram bins
        self.spatial_feat = False  # Spatial features on or off
        self.hist_feat = True  # Histogram features on or off
        self.hog_feat = False  # HOG features on or off        

        filename = 'finalized_model.sav'
        filename_scaler = 'scalar.sav'
        if os.path.isfile(filename) and os.path.isfile(filename_scaler):
            # Load trained model from disk
            rospy.loginfo("TLClassifier: loading model from %s ...", filename)
            self.svc = pickle.load(open(filename, 'rb'))
            rospy.loginfo("TLClassifier: loading scaler from %s ...", filename_scaler)
            self.X_scaler = pickle.load(open(filename_scaler, 'rb'))

        else:
            # Import traffic light dataset for training
            redlights = []
            nonredlights = []
            # Read in redlight samples
            rospy.loginfo("TLClassifier: importing dataset ...")
            images_red = glob.glob("dataset/sim/red/*.png")
            for image in images_red:
                redlights.append(image)
            # Read in non-redlight samples
            images_notred = glob.glob("dataset/sim/notred/*.png")
            for image in images_notred:
                nonredlights.append(image)
            # Extract features using above defined parameters
            rospy.loginfo("TLClassifier: extracting features...")
            redlights_features = extract_features(redlights, color_space=self.color_space,
                                                  spatial_size=self.spatial_size, hist_bins=self.hist_bins,
                                                  orient=self.orient, pix_per_cell=self.pix_per_cell,
                                                  cell_per_block=self.cell_per_block,
                                                  hog_channel=self.hog_channel, spatial_feat=self.spatial_feat,
                                                  hist_feat=self.hist_feat, hog_feat=self.hog_feat)
            nonredlights_features = extract_features(nonredlights, color_space=self.color_space,
                                                     spatial_size=self.spatial_size, hist_bins=self.hist_bins,
                                                     orient=self.orient, pix_per_cell=self.pix_per_cell,
                                                     cell_per_block=self.cell_per_block,
                                                     hog_channel=self.hog_channel, spatial_feat=self.spatial_feat,
                                                     hist_feat=self.hist_feat, hog_feat=self.hog_feat)

            # Prepare datasets for training
            rospy.loginfo("TLClassifier: preparing datasets for training...")
            X = np.vstack((redlights_features, nonredlights_features)).astype(np.float64)
            # Fit a per-column scaler
            self.X_scaler = StandardScaler().fit(X)
            # Apply the scaler to X
            scaled_X = self.X_scaler.transform(X)
            # Define the labels vector
            y = np.hstack((np.ones(len(redlights_features)), np.zeros(len(nonredlights_features))))
            # Split up data into randomized training and test sets
            rand_state = np.random.randint(0, 100)
            X_train, X_test, y_train, y_test = train_test_split(
                scaled_X, y, test_size=0.2, random_state=rand_state)
            rospy.loginfo("TLClassifier: feature vector length is %i", len(X_train[0]))
            rospy.loginfo("TLClassifier: total length is %i", len(X_train))

            # Create and train classifier
            rospy.loginfo("TLClassifier: creating and training SVC...")
            # Use a linear SVC
            self.svc = LinearSVC()
            # Check the training time for the SVC
            t = time.time()
            self.svc.fit(X_train, y_train)
            t2 = time.time()
            duration = round(t2 - t, 2)
            rospy.loginfo("TLClassifier: %.2f seconds to train SVC", duration)
            # Check the score of the SVC
            score = round(self.svc.score(X_test, y_test))
            rospy.loginfo("TLClassifier: Test Accuracy of SVC = %.2f", score)
            # Save the model to disk
            rospy.loginfo("TLClassifier: saving model... ")
            pickle.dump(self.svc, open(filename, 'wb'))
            rospy.loginfo("TLClassifier: saving scaler... ")
            pickle.dump(self.X_scaler, open(filename_scaler, 'wb'))

    def get_classification(self, input_image):
        """Determines the color of the traffic light in the image
        Args:
            input_image (cv::Mat): image containing the traffic light
        Returns:
            tl_state (int): ID of traffic light color (specified in styx_msgs/TrafficLight)
        """

        # Classify input image
        prediction = search_windows(input_image, self.svc, self.X_scaler, color_space=self.color_space,
                                        spatial_size=self.spatial_size, hist_bins=self.hist_bins,
                                        orient=self.orient, pix_per_cell=self.pix_per_cell,
                                        cell_per_block=self.cell_per_block,
                                        hog_channel=self.hog_channel, spatial_feat=self.spatial_feat,
                                        hist_feat=self.hist_feat, hog_feat=self.hog_feat)
        tl_state = 4  # TrafficLight.UNKNOWN
        if prediction == 1:
            # Red traffic light predicted
            tl_state = 0  # TrafficLight.RED
        elif prediction == 0:
            # Non-Red traffic light predicted
            tl_state = 2  # TrafficLight.GREEN

        return tl_state
