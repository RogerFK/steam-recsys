import normalization
import pandas as pd
data = pd.DataFrame({
        "steamid":          [   1,   1,    1,    1,     2,  2,   2,   2],
        "appid":            [   1,   2,    3,    4,     1,  2,   3,   4],
        "playtime_forever": [6000, 120, 3000, 3000, 10000, 20, 120, 240]
    })
norm = normalization.LinearPlaytimeNormalizer()
def test_normalize():
    data = pd.DataFrame({
        "steamid":          [1,      1,    1,    1,     2,  2,   2,   2],
        "appid":            [1, 2, 3, 4, 1, 2, 3, 4],
        "playtime_forever": [6000, 120, 3000, 3000, 10000, 20, 120, 240]
    })
    # note 20 = 60
    expected = pd.DataFrame({
        "steamid": [1, 1, 1, 1, 2, 2, 2, 2],
        "appid": [1, 2, 3, 4, 1, 2, 3, 4],
        "playtime_forever": [1, 0.02, 0.5, 0.5, 1, 0.006, 0.012, 0.024]
    })
    print("Input\n", data)
    normalized = norm.normalize(data)
    print("Input again\n", data)
    print("Output\n", normalized)
    print("Expected\n", expected)
    
    assert normalized.equals(expected)
    import timeit
    print("Timing")
    print(timeit.timeit("norm.normalize(data)", setup="from normalization_tests import norm, data", number=1000))

if __name__ == "__main__":
    test_normalize()