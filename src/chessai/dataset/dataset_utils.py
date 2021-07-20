
import tensorflow as tf
import numpy as np


def convert_states(states: np.ndarray) -> np.ndarray:

    num_samples = states.shape[0]

    # convert uint64 bitboards to 64 bits as float32 each -> (samples x 13 x 64)
    bitboard_bits = np.unpackbits(np.array(states, dtype='>i8').view(np.uint8))
    bitboard_bits = np.reshape(bitboard_bits, (num_samples, 13, 64)).astype(np.float32)

    # transpose the bitboards -> (samples x 8 x 8 x 13)
    bitboards_reshaped = np.transpose(bitboard_bits, [0, 2, 1])
    bitboards_reshaped = np.reshape(bitboards_reshaped, (num_samples, 8, 8, 13))
    return bitboards_reshaped


# def convert_states(states: np.ndarray):

#     num_samples = states.shape[0]

#     # convert uint64 bitboards to 64 bits as float32 each -> (samples x 13 x 64)
#     bitboard_bits = np.unpackbits(np.array(states, dtype='>i8').view(np.uint8))
#     bitboard_bits = np.reshape(bitboard_bits, (num_samples, 13, 64)).astype(np.float32)

#     # split bitboards into categories -> 2x (samples x 6 x 64), 1x (samples x 64)
#     bitboard_bits_white = bitboard_bits[:, 0:6, :]
#     bitboard_bits_black = bitboard_bits[:, 6:12, :]
#     bitboard_bits_wasmoved = bitboard_bits[:, 12, :]

#     # aggregate the white and black bitboards (white=1, black=-1, nothing=0)
#     bitboards_compressed = np.zeros((num_samples, 7, 64), dtype=np.float32)
#     bitboards_compressed[:, 0:6, :] = bitboard_bits_white - bitboard_bits_black
#     bitboards_compressed[:, 6, :] = bitboard_bits_wasmoved

#     # transpose the bitboards -> (samples x 8 x 8 x 7)
#     bitboards_reshaped = np.transpose(bitboards_compressed, [0, 2, 1])
#     bitboards_reshaped = np.reshape(bitboards_reshaped, (num_samples, 8, 8, 7))
#     return bitboards_reshaped


# @tf.function
# def chessboard_to_compact_2d_feature_maps(bitboards):

#     # extract each bit from the bitboards -> bring it into 13x8x8 shape
#     feature_maps = chessboard_to_feature_maps(bitboards)

#     # union the pieces of each type onto a single feature map and encode it
#     # as follows: white piece = 1, black piece = -1, nothing = 0
#     # the resulting maps are of shape 6x8x8
#     unified_piece_maps = feature_maps[0:6, :, :] + (feature_maps[6:12, :, :] * -1.0)

#     # append the was_moved bitboard to piece maps -> 6x8x8 shape
#     compressed_feature_maps = tf.concat((unified_piece_maps, feature_maps[12:13, :, :]), axis=0)

#     # transpose the feature maps, so the content can be convoluted
#     # by field positions and maps are the channels -> shape 8x8x7
#     transp_feature_maps = tf.transpose(compressed_feature_maps, (1, 2, 0))
#     return transp_feature_maps


# @tf.function
# def chessboard_to_feature_maps(bitboards):

#     # create empty tensor array for bitboards
#     feature_maps = tf.TensorArray(dtype=tf.float32, size=13, dynamic_size=False)

#     # loop through all bitboards
#     for i in tf.range(tf.size(bitboards)):

#         # load single bitboard and create a feature map to write to
#         temp_bitboard = bitboards[i]
#         temp_feature_map = tf.TensorArray(dtype=tf.float32, size=64, dynamic_size=False)

#         # loop through all bitboard positions
#         for pos in tf.range(64):
#             # extract the piece_set bit from the bitboard: (bitboard >> pos) & 1
#             bit = tf.bitwise.bitwise_and(tf.bitwise.right_shift(temp_bitboard, tf.cast(pos, dtype=tf.uint64)), 1)
#             temp_feature_map = temp_feature_map.write(pos, tf.cast(bit, tf.float32))

#         # reshape the 64 bitboard positions to 8x8 format, dimensions are (row, column)
#         temp_feature_map = temp_feature_map.stack()
#         temp_feature_map = tf.reshape(temp_feature_map, (8, 8))

#         # apply the converted feature map to the output
#         feature_maps = feature_maps.write(i, temp_feature_map)

#     feature_maps = feature_maps.stack()
#     return feature_maps