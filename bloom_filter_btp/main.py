import math
import os

import numpy as np
import cv2 as cv
from deepface import DeepFace
from img_io import capture_image

from icecream import ic

DBdir = "./data/images/"
DB_BFtemplates = "./data/templates/"
# Parameters of Bloom filter extraction
nXORKeys = 0  # number of XOR keys to use (0 for no XOR)
N_BITS_BF = 10
N_WORDS_BF = 32
N_BF_X = 512 // N_WORDS_BF
N_BF_Y = 20 // N_BITS_BF
N_BLOCKS = N_BF_X * N_BF_Y
BF_SIZE = int(math.pow(2, N_BITS_BF))
THRESHOLD = 0
N_HIST = 40  # parameters fixed by LGBPHS
N_BINS = 59
cropped_face_size = (80, 64)

FEATURE_SIZE = (N_WORDS_BF * N_BF_X, N_BITS_BF * N_BF_Y)

class LGBPHS:
    def __init__(self, block_size, block_overlap, gabor_directions=8, gabor_scales=5, use_gabor_phases=False,
                 lbp_radius=2, lbp_neighbor_count=8):
        self.block_size = block_size
        self.block_overlap = block_overlap
        self.gabor_directions = gabor_directions
        self.gabor_scales = gabor_scales
        self.use_gabor_phases = use_gabor_phases
        self.lbp_radius = lbp_radius
        self.lbp_neighbor_count = lbp_neighbor_count

    def gabor_kernels(self):
        kernels = []
        for scale in range(self.gabor_scales):
            for direction in range(self.gabor_directions):
                theta = direction * np.pi / self.gabor_directions
                sigma = 1.0
                lamda = np.pi * (scale + 1) / self.gabor_scales
                gamma = 1.0
                psi = 0.0
                kernel = cv.getGaborKernel((21, 21), sigma, theta, lamda, gamma, psi)
                kernels.append(kernel)
        return kernels

    def lbp_histogram(self, image):
        lbp_image = cv.calcHist([image], [0], None, [N_BINS], ranges=[0, 256])
        return lbp_image

    def extract_blocks(self, image):
        h, w = image.shape
        block_h, block_w = self.block_size
        overlap_h, overlap_w = self.block_overlap

        blocks = []

        y = 0
        while y < h:
            x = 0
            while x < w:
                block = image[y:y+block_h, x:x+block_w]
                if block.shape[0] == block_h and block.shape[1] == block_w:
                    blocks.append(block)
                x += (block_w - overlap_w)
            y += (block_h - overlap_h)

        return blocks

    def extract_features(self, image):
        kernels = self.gabor_kernels()

        features = []
        for kernel in kernels:
            transformed = cv.filter2D(image, cv.CV_32F, kernel)
            blocks = self.extract_blocks(transformed)
            for block in blocks:
                hist_abs = self.lbp_histogram(np.abs(block))
                features.append(hist_abs)
                if self.use_gabor_phases:
                    hist_phase = self.lbp_histogram(np.angle(block))
                    features.append(hist_phase)

        return np.array(features)

    def __call__(self, image):


        # convert to grayscale
        features = self.extract_features(image)

        # remove one dimension
        features = np.squeeze(features)


        # extract blocks
        features = self.extract_blocks(features)


        return features



def add_unlinkability(features, keyPERM):
    """Permutes rows within regions of an iris-code to achieve unlinkability"""
    perm_feat = np.zeros(shape=features.shape, dtype=int)

    # divide iris-code in four regions, and reshape each region to a size [N_BITS_BF * N_BF_X / 2, N_WORDS_BF]
    featsAux1 = np.reshape(
        features[0:N_BITS_BF, 0 : (N_BF_X / 2) * N_WORDS_BF],
        [N_BITS_BF * N_BF_X / 2, N_WORDS_BF],
    )
    featsAux2 = np.reshape(
        features[0:N_BITS_BF, (N_BF_X / 2) * N_WORDS_BF : N_BF_X * N_WORDS_BF],
        [N_BITS_BF * N_BF_X / 2, N_WORDS_BF],
    )
    featsAux3 = np.reshape(
        features[N_BITS_BF : 2 * N_BITS_BF, 0 : (N_BF_X / 2) * N_WORDS_BF],
        [N_BITS_BF * N_BF_X / 2, N_WORDS_BF],
    )
    featsAux4 = np.reshape(
        features[
            N_BITS_BF : 2 * N_BITS_BF, (N_BF_X / 2) * N_WORDS_BF : N_BF_X * N_WORDS_BF
        ],
        [N_BITS_BF * N_BF_X / 2, N_WORDS_BF],
    )

    # permute rows within each region
    perm_feat[0:N_BITS_BF, 0 : (N_BF_X / 2) * N_WORDS_BF] = np.reshape(
        featsAux1[keyPERM[0, :], :], [N_BITS_BF, (N_BF_X / 2) * N_WORDS_BF]
    )
    perm_feat[
        0:N_BITS_BF, (N_BF_X / 2) * N_WORDS_BF : N_BF_X * N_WORDS_BF
    ] = np.reshape(
        featsAux2[keyPERM[2, :], :], [N_BITS_BF, (N_BF_X / 2) * N_WORDS_BF]
    )
    perm_feat[N_BITS_BF : 2 * N_BITS_BF, 0 : (N_BF_X / 2) * N_WORDS_BF] = np.reshape(
        featsAux3[keyPERM[2, :], :], [N_BITS_BF, (N_BF_X / 2) * N_WORDS_BF]
    )
    perm_feat[
        N_BITS_BF : 2 * N_BITS_BF, (N_BF_X / 2) * N_WORDS_BF : N_BF_X * N_WORDS_BF
    ] = np.reshape(
        featsAux4[keyPERM[3, :], :], [N_BITS_BF, (N_BF_X / 2) * N_WORDS_BF]
    )

    return perm_feat


def hamming_distance(X, Y):
    """Computes the noralised Hamming distance between two Bloom filter templates"""
    dist = 0

    X = np.array(X, dtype=int)
    Y = np.array(Y, dtype=int)

    N_BLOCKS = X.shape[0]
    for i in range(N_BLOCKS):
        A = X[i, :]
        B = Y[i, :]

        suma = sum(A) + sum(B)
        if suma > 0:
            dist += float(sum(A ^ B)) / float(suma)

    return dist / float(N_BLOCKS)


def gen_bf_from_face(feat):
    '''Extracts BF protected template from an unprotected template'''
    template = []

    index = 0
    for block in feat:

        #change to int
        block = np.array(block, dtype=int)

        # binarize
        block = np.where(block > 0, 1, 0)

        bf = np.zeros(shape=[BF_SIZE])


        for k in range(block.shape[1]):
            hist = block[:, k]
            location = int("0b" + ''.join([str(a) for a in hist]), 2)
            bf[location] = int(1)

        template.append(bf)

    return template


def register_user():
    # get user name
    username = input("Enter your name: ")

    # get user face
    print("Please, look at the camera...")
    img = capture_image()


    # detect and crop face with opencv
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    if len(faces) == 0:
        print("No face detected")
        return
    x, y, w, h = faces[0]
    img = img[y:y+h, x:x+w]
    img = cv.resize(img, cropped_face_size)


    # extract face features
    print("Extracting face features...")
    face = LGBPHS((N_BITS_BF, N_WORDS_BF), (0,0))(img)

    # generate user bloom filter
    print("Generating user BF...")
    bf = np.array(gen_bf_from_face(face))
    

    # save user bloom filter
    print("Saving user BF...")
    np.savetxt(DB_BFtemplates + username + "_BFtemplate.txt", bf, fmt="%d")

    print("User registered successfully!")


def check_user_in_db(threshold: float) -> bool:
    # get user name

    # get user face
    print("Please, look at the camera...")
    img = capture_image()
    # detect and crop face with opencv
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    if len(faces) == 0:
        print("No face detected")
        return False
    x, y, w, h = faces[0]
    img = img[y:y+h, x:x+w]
    img = cv.resize(img, cropped_face_size)



    # extract face features
    print("Extracting face features...")
    face = LGBPHS((N_BITS_BF, N_WORDS_BF), (0,0))(img)


    # generate user bloom filters
    print("Generating user BF...")
    bf = gen_bf_from_face(face)

    # load all bloom filters
    print("Loading all BF templates...")
    bf_templates = [] 

    for file in os.listdir(DB_BFtemplates):
        
        bf_templates.append(np.loadtxt(DB_BFtemplates + file, dtype=int))

    users = os.listdir(DB_BFtemplates)
    
    # check if user is in the database
    print("Checking if user is in the database...")
    for i, t in enumerate(bf_templates):
        dist = hamming_distance(bf, t)
        print("Distance with user " + users[i]  + ": " + str(dist))

    print("User not found!")
    return False





def main() -> None:
    while True:
        print("1. Register user")
        print("2. Login user")
        print("3. Exit")
        option = input("Select an option: ")

        if option == "1":
            register_user()
        elif option == "2":
            check_user_in_db(0.5)
        elif option == "3":
            print("Bye!")
            break
        else:
            print("Invalid option")
            continue

    # Define permutation key to provide unlinkability
    key = np.zeros(shape=[4, N_BITS_BF * N_BF_X / 2], dtype=int)
    for j in range(4):
        key[j, :] = np.random.permutation(N_BITS_BF * N_BF_X / 2)


if __name__ == "__main__":
    main()
