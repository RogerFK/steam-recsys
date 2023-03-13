import normalization
import pandas as pd
from pandas.testing import assert_frame_equal
import timeit
import os

data = pd.DataFrame({
        "steamid":          [   1,   1,    1,    1,     2,  2,   2,   2],
        "appid":            [   1,   2,    3,    4,     1,  2,   3,   4],
        "playtime_forever": [6000, 120, 3000, 3000, 10000, 20, 120, 240]
    })
# note 20 = 60
linear_norm_max = normalization.LinearPlaytimeNormalizer('max')
log_norm_max = normalization.LogPlaytimeNormalizer('max')
root_norm_max = normalization.RootPlaytimeNormalizer('max')

linear_norm_sum = normalization.LinearPlaytimeNormalizer('sum')
log_norm_sum = normalization.LogPlaytimeNormalizer('sum')
root_norm_sum = normalization.RootPlaytimeNormalizer('sum')

def test_normalize():
    if not os.path.exists("./test_output"):
        os.mkdir("./test_output")
    print("Input\n", data)

    print ("----------------\nUsing max normalization\n----------------")

    linear_expected_max = pd.read_csv("./test_expected/linear_normalized_max.csv")
    print("Linear output\n", linear_norm_max.normalize(data))
    normalized = linear_norm_max.normalize(data)
    normalized.to_csv("./test_output/linear_normalized_max.csv", index=False)
    assert_frame_equal(normalized, linear_expected_max)
    print("Timing for linear normalization (max)")
    print(timeit.timeit("linear_norm_max.normalize(data)", setup="from normalization_tests import linear_norm_max, data", number=1000))

    log_expected_max = pd.read_csv("./test_expected/log_normalized_max.csv")
    normalized = log_norm_max.normalize(data)
    print("Log output\n", normalized)
    normalized.to_csv("./test_output/log_normalized_max.csv", index=False)
    print(log_expected_max)
    assert_frame_equal(normalized, log_expected_max)
    print("Timing for log normalization (max)")
    print(timeit.timeit("log_norm_max.normalize(data)", setup="from normalization_tests import log_norm_max, data", number=1000))

    root_expected_max = pd.read_csv("./test_expected/root_normalized_max.csv")
    normalized = root_norm_max.normalize(data)
    print("Root output\n", normalized)
    normalized.to_csv("./test_output/root_normalized_max.csv", index=False)
    assert_frame_equal(normalized, root_expected_max)
    print("Timing for root normalization (max)")
    print(timeit.timeit("root_norm_max.normalize(data)", setup="from normalization_tests import root_norm_max, data", number=1000))
    
    print ("----------------\nUsing sum normalization\n----------------")
    
    linear_expected_sum = pd.read_csv("./test_expected/linear_normalized_sum.csv")
    normalized = linear_norm_sum.normalize(data)
    print("Linear output\n", normalized)
    normalized.to_csv("./test_output/linear_normalized_sum.csv", index=False)
    assert_frame_equal(normalized, linear_expected_sum)
    print("Timing for linear normalization (sum)")
    print(timeit.timeit("linear_norm_sum.normalize(data)", setup="from normalization_tests import linear_norm_sum, data", number=1000))

    log_expected_sum = pd.read_csv("./test_expected/log_normalized_sum.csv")
    normalized = log_norm_sum.normalize(data)
    print("Log output\n", normalized)
    normalized.to_csv("./test_output/log_normalized_sum.csv", index=False)
    assert_frame_equal(normalized, log_expected_sum)
    print("Timing for log normalization (sum)")
    print(timeit.timeit("log_norm_sum.normalize(data)", setup="from normalization_tests import log_norm_sum, data", number=1000))

    root_expected_sum = pd.read_csv("./test_expected/root_normalized_sum.csv")
    normalized = root_norm_sum.normalize(data)
    print("Root output\n", normalized)
    normalized.to_csv("./test_output/root_normalized_sum.csv", index=False)
    assert_frame_equal(normalized, root_expected_sum)
    print("Timing for root normalization (sum)")
    print(timeit.timeit("root_norm_sum.normalize(data)", setup="from normalization_tests import root_norm_sum, data", number=1000))          

if __name__ == "__main__":
    test_normalize()
    print("All tests passed")