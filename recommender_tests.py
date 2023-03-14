from recommender import *
from normalization import *

rand = RandomRecommenderSystem()

def test_recommendations_from_priority_queue():
    pq = PriorityQueue()
    pq.put((1, 1))
    pq.put((2, 2))
    pq.put((3, 3))
    df = rand.recommendations_from_priority_queue(pq)
    assert df.shape == (3, 2)
    assert df.iloc[0]["appid"] == 3
    assert df.iloc[1]["appid"] == 2
    assert df.iloc[2]["appid"] == 1

def test_recommend():
    data = pd.DataFrame(
        [
            {"steamid": 1, "appid": 1, "playtime_forever": 1},
            {"steamid": 1, "appid": 2, "playtime_forever": 2},
            {"steamid": 1, "appid": 3, "playtime_forever": 3},
            {"steamid": 1, "appid": 4, "playtime_forever": 4},
            {"steamid": 1, "appid": 5, "playtime_forever": 5},
        ]
    )
    df = rand.recommend(data, 1, n=3)
    assert df.shape == (3, 2)
    assert df.iloc[0]["appid"] in [1, 2, 3, 4, 5]
    assert df.iloc[1]["appid"] in [1, 2, 3, 4, 5]
    assert df.iloc[2]["appid"] in [1, 2, 3, 4, 5]