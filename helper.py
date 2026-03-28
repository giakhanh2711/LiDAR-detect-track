import numpy as np

HUMAN_SPEED  = 1.4         # m/s
MAX_ASSOCIATION_DISTANCE = 0.5  # meters, max distance for track association
HUMAN_SIZE = []
INITIAL_NUM_SCAN = 3
THRESHOLD_DETECT_WALL = 0.2

QOS_PROFILE = 10
PUBLISH_MARKER_TIME_PERIOD = 0.1
INITIAL_COV = np.identity(4) / 10.0

DELTA_T = 0.1

# TODO: Define A, B, G, C, H for KF
A = np.array([[1, 0, DELTA_T, 0],
              [0, 1, 0, DELTA_T],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

B = np.identity(n=4)
G = np.identity(n=4)
C = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])
H = np.identity(n=2)

SIGMA_THETA = np.identity(n=4)
SIGMA_PSI = np.identity(n=2)
    
def ranges_to_xy(range, theta):

    """
    return np.array([[x1, y1], [x2, y2],...])
    """
    x = range * np.cos(theta)
    y = range * np.sin(theta)
    return np.column_stack((x, y))


