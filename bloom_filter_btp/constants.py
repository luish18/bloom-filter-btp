# image resolution of the preprocessed images
CROPPED_IMAGE_HEIGHT = 80
CROPPED_IMAGE_WIDTH = 64

# eye positions for frontal images
RIGHT_EYE_POS = (CROPPED_IMAGE_HEIGHT / 4, CROPPED_IMAGE_WIDTH / 4 - 1)
LEFT_EYE_POS  = (CROPPED_IMAGE_HEIGHT / 4, CROPPED_IMAGE_WIDTH / 4 * 3)

# Parameters of LGBPHS and Bloom filter extraction
N_BLOCKS = (
    80  # number of blocks the facial image is divided into, also for LGBPHS algorihtm
)

N_HIST = 40  # parameters fixed by LGBPHS
N_BINS = 59

THRESHOLD = 0  # binarization threshold for LGBPHS features

N_BITS_BF = 4  # parameters for BF extraction
N_WORDS_BF = 20
BF_SIZE = 2**N_BITS_BF
N_BF_Y = N_HIST // N_BITS_BF
N_BF_X = (N_BINS + 1) // N_WORDS_BF


