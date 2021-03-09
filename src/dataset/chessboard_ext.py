
import tensorflow as tf


@tf.function
def chessboard_to_compact_2d_feature_maps(bitboards):

    # extract each bit from the bitboards -> bring it into 13x8x8 shape
    feature_maps = chessboard_to_feature_maps(bitboards)

    # union the pieces of each type onto a single feature map and encode it
    # as follows: white piece = 1, black piece = -1, nothing = 0
    # the resulting maps are of shape 6x8x8
    unified_piece_maps = feature_maps[0:6, :, :] + (feature_maps[6:12, :, :] * -1.0)

    # append the was_moved bitboard to piece maps -> 6x8x8 shape
    compressed_feature_maps = tf.concat((unified_piece_maps, feature_maps[12:13, :, :]), axis=0)

    # transpose the feature maps, so the content can be convoluted
    # by field positions and maps are the channels -> shape 8x8x7
    transp_feature_maps = tf.transpose(compressed_feature_maps, (1, 2, 0))
    return transp_feature_maps


@tf.function
def chessboard_to_feature_maps(bitboards):

    # create empty tensor array for bitboards
    feature_maps = tf.TensorArray(dtype=tf.float32, size=13, dynamic_size=False)

    # loop through all bitboards
    for i in tf.range(tf.size(bitboards)):

        # load single bitboard and create a feature map to write to
        temp_bitboard = bitboards[i]
        temp_feature_map = tf.TensorArray(dtype=tf.float32, size=64, dynamic_size=False)

        # loop through all bitboard positions
        for pos in tf.range(64):
            # extract the piece_set bit from the bitboard: (bitboard >> pos) & 1
            bit = tf.bitwise.bitwise_and(tf.bitwise.right_shift(temp_bitboard, tf.cast(pos, dtype=tf.uint64)), 1)
            temp_feature_map = temp_feature_map.write(pos, tf.cast(bit, tf.float32))

        # reshape the 64 bitboard positions to 8x8 format, dimensions are (row, column)
        temp_feature_map = temp_feature_map.stack()
        temp_feature_map = tf.reshape(temp_feature_map, (8, 8))

        # apply the converted feature map to the output
        feature_maps = feature_maps.write(i, temp_feature_map)

    feature_maps = feature_maps.stack()
    return feature_maps