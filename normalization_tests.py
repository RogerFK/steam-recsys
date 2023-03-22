import normalization
import pandas as pd
from pandas.testing import assert_frame_equal
import timeit
import os
import logging

data = pd.DataFrame({
        "steamid":          [   1,   1,    1,    1,     2,  2,   2,   2],
        "appid":            [   1,   2,    3,    4,     1,  2,   3,   4],
        "playtime_forever": [6000, 120, 3000, 3000, 10000, 20, 120, 240] # note any playtime below 60 will be 60
    })

TIMEIT_ITERATIONS = 100
OUTPUT_MULTIPLIER = 5
DEBUG = False
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO, format='%(asctime)s - %(message)s')

linear_norm_max = normalization.LinearPlaytimeNormalizer('max', output_multiplier=OUTPUT_MULTIPLIER)
log_norm_max = normalization.LogPlaytimeNormalizer('max', output_multiplier=OUTPUT_MULTIPLIER)
root_norm_max = normalization.RootPlaytimeNormalizer('max', output_multiplier=OUTPUT_MULTIPLIER)

linear_norm_sum = normalization.LinearPlaytimeNormalizer('sum', output_multiplier=OUTPUT_MULTIPLIER)
log_norm_sum = normalization.LogPlaytimeNormalizer('sum', output_multiplier=OUTPUT_MULTIPLIER)
root_norm_sum = normalization.RootPlaytimeNormalizer('sum', output_multiplier=OUTPUT_MULTIPLIER)

linear_norm_sum_max = normalization.LinearPlaytimeNormalizer('sum_max', output_multiplier=OUTPUT_MULTIPLIER)
log_norm_sum_max = normalization.LogPlaytimeNormalizer('sum_max', output_multiplier=OUTPUT_MULTIPLIER)
root_norm_sum_max = normalization.RootPlaytimeNormalizer('sum_max', output_multiplier=OUTPUT_MULTIPLIER)


def test_normalize():
    if not os.path.exists("./test_output"):
        os.mkdir("./test_output")
    logging.debug("Input\n", data)

    logging.debug("----------------\nUsing max normalization\n----------------")
    try:
        linear_expected_max = pd.read_csv(f"./test_expected_{OUTPUT_MULTIPLIER}/linear_normalized_max.csv")
    except FileNotFoundError:
        linear_expected_max = None
    logging.debug("Linear output\n", linear_norm_max.normalize(data))
    normalized = linear_norm_max.normalize(data)
    normalized.to_csv("./test_output/linear_normalized_max.csv", index=False)
    if linear_expected_max is not None: assert_frame_equal(normalized, linear_expected_max)
    logging.info(f"Timing for linear normalization (max) {repr(linear_norm_max)}: " + str(timeit.timeit("linear_norm_max.normalize(data)", setup="from normalization_tests import linear_norm_max, data", number=TIMEIT_ITERATIONS)))

    try:
        log_expected_max = pd.read_csv(f"./test_expected_{OUTPUT_MULTIPLIER}/log_normalized_max.csv")
    except FileNotFoundError:
        log_expected_max = None
    normalized = log_norm_max.normalize(data)
    logging.debug("Log output\n", normalized)
    normalized.to_csv("./test_output/log_normalized_max.csv", index=False)
    logging.debug(log_expected_max)
    if log_expected_max is not None: assert_frame_equal(normalized, log_expected_max)
    logging.info("Timing for log normalization (max): " + str(timeit.timeit("log_norm_max.normalize(data)", setup="from normalization_tests import log_norm_max, data", number=TIMEIT_ITERATIONS)))

    try:
        root_expected_max = pd.read_csv(f"./test_expected_{OUTPUT_MULTIPLIER}/root_normalized_max.csv")
    except FileNotFoundError:
        root_expected_max = None
    normalized = root_norm_max.normalize(data)
    logging.debug("Root output\n", normalized)
    normalized.to_csv("./test_output/root_normalized_max.csv", index=False)
    if root_expected_max is not None: assert_frame_equal(normalized, root_expected_max)
    logging.info("Timing for root normalization (max): " + str(timeit.timeit("root_norm_max.normalize(data)", setup="from normalization_tests import root_norm_max, data", number=TIMEIT_ITERATIONS)))
    
    logging.debug ("----------------\nUsing sum normalization\n----------------")
    
    try:
        linear_expected_sum = pd.read_csv(f"./test_expected_{OUTPUT_MULTIPLIER}/linear_normalized_sum.csv")
    except FileNotFoundError:
        linear_expected_sum = None
    normalized = linear_norm_sum.normalize(data)
    logging.debug("Linear output\n", normalized)
    normalized.to_csv("./test_output/linear_normalized_sum.csv", index=False)
    if linear_expected_sum is not None: assert_frame_equal(normalized, linear_expected_sum)
    logging.info("Timing for linear normalization (sum): " + str(timeit.timeit("linear_norm_sum.normalize(data)", setup="from normalization_tests import linear_norm_sum, data", number=TIMEIT_ITERATIONS)))

    try:
        log_expected_sum = pd.read_csv(f"./test_expected_{OUTPUT_MULTIPLIER}/log_normalized_sum.csv")
    except FileNotFoundError:
        log_expected_sum = None
    normalized = log_norm_sum.normalize(data)
    logging.debug("Log output\n", normalized)
    normalized.to_csv("./test_output/log_normalized_sum.csv", index=False)
    if log_expected_sum is not None: assert_frame_equal(normalized, log_expected_sum)
    logging.info("Timing for log normalization (sum): " + str(timeit.timeit("log_norm_sum.normalize(data)", setup="from normalization_tests import log_norm_sum, data", number=TIMEIT_ITERATIONS)))

    try:
        root_expected_sum = pd.read_csv(f"./test_expected_{OUTPUT_MULTIPLIER}/root_normalized_sum.csv")
    except FileNotFoundError:
        root_expected_sum = None
    normalized = root_norm_sum.normalize(data)
    logging.debug("Root output\n", normalized)
    normalized.to_csv("./test_output/root_normalized_sum.csv", index=False)
    if root_expected_sum is not None: assert_frame_equal(normalized, root_expected_sum)
    logging.info("Timing for root normalization (sum): " + str(timeit.timeit("root_norm_sum.normalize(data)", setup="from normalization_tests import root_norm_sum, data", number=TIMEIT_ITERATIONS)))          

    logging.debug ("----------------\nUsing sum_max normalization\n----------------")
    try:
        linear_expected_sum_max = pd.read_csv(f"./test_expected_{OUTPUT_MULTIPLIER}/linear_normalized_sum_max.csv")
    except:
        linear_expected_sum_max = None
    normalized = linear_norm_sum_max.normalize(data)
    logging.debug("Linear output\n", normalized)
    normalized.to_csv("./test_output/linear_normalized_sum_max.csv", index=False)
    if linear_expected_sum_max is not None:
        assert_frame_equal(normalized, linear_expected_sum_max)
    logging.info("Timing for linear normalization (sum_max): " + str(timeit.timeit("linear_norm_sum_max.normalize(data)", setup="from normalization_tests import linear_norm_sum_max, data", number=TIMEIT_ITERATIONS)))

    try:
        log_expected_sum_max = pd.read_csv(f"./test_expected_{OUTPUT_MULTIPLIER}/log_normalized_sum_max.csv")
    except FileNotFoundError:
        log_expected_sum_max = None
    normalized = log_norm_sum_max.normalize(data)
    logging.debug("Log output\n", normalized)
    normalized.to_csv("./test_output/log_normalized_sum_max.csv", index=False)
    if log_expected_sum_max is not None:
        assert_frame_equal(normalized, log_expected_sum_max)
    logging.info("Timing for log normalization (sum_max): " + str(timeit.timeit("log_norm_sum_max.normalize(data)", setup="from normalization_tests import log_norm_sum_max, data", number=TIMEIT_ITERATIONS)))
    try:
        root_expected_sum_max = pd.read_csv(f"./test_expected_{OUTPUT_MULTIPLIER}/root_normalized_sum_max.csv")
    except FileNotFoundError:
        root_expected_sum_max = None
    normalized = root_norm_sum_max.normalize(data)
    logging.debug("Root output\n", normalized)
    normalized.to_csv("./test_output/root_normalized_sum_max.csv", index=False)
    if root_expected_sum_max is not None:
        assert_frame_equal(normalized, root_expected_sum_max)
    logging.info("Timing for root normalization (sum_max): " + str(timeit.timeit("root_norm_sum_max.normalize(data)", setup="from normalization_tests import root_norm_sum_max, data", number=TIMEIT_ITERATIONS)))

    if not os.path.exists(f"./test_expected_{OUTPUT_MULTIPLIER}"):
        os.mkdir(f"./test_expected_{OUTPUT_MULTIPLIER}")
        logging.info("Created directory for expected output, if you believe the results are correct feel free to copy the output to the expected folder for future tests.")

if __name__ == "__main__":
    test_normalize()
    logging.info("All tests passed")