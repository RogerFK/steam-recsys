import argparse
import os
import random
import shutil
import sys
import time

import numpy as np
import pandas as pd

import normalization
import recommender

import matplotlib.pyplot as plt

import concurrent.futures as cf
import signal

import gc as garbage_collector

# Our goal is to explore which thresholds, mix of normalization methods and recommenders/mix of recommenders work best.
# This script assumes you have already run the split_train_test.py script and your data is inside the data/ folder.
# If the data isn't there, we will split the data and save it to the data/ folder using the default values.
import split_train_test

import logging
import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    filename=f"output {datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.log", filemode='w'
                    )


# change BIN_DATA_PATH to "bin_data_test" to avoid conflicts
recommender.BIN_DATA_PATH = "experiments/bin_data/"
result_path = "experiments/results/"
os.makedirs(recommender.BIN_DATA_PATH, exist_ok=True)
os.makedirs(result_path, exist_ok=True)

def recommend_user(recommender_system: recommender.AbstractRecommenderSystem, recommender_name, steamid: int):
    # beforehand, check if there's results already
    results_file = os.path.join(result_path, recommender_name, f"{steamid}_results.csv")
    if os.path.exists(results_file):
        return (steamid, pd.read_csv(results_file))
    results = recommender_system.recommend(steamid, n=50, filter_owned=True)
    # results_list.append((steamid, results))
    # store the results, which are a DataFrame (or if they're a list of tuples, we can convert it to a DataFrame)
    if not os.path.exists(os.path.join(result_path, recommender_name)):
        os.mkdir(os.path.join(result_path, recommender_name))
    if isinstance(results, list):
        results = pd.DataFrame(results, columns=["appid", "score"])
    results.to_csv(os.path.join(result_path, recommender_name, f"{steamid}_results.csv"), index=True)
    print(f"Recommender system {recommender_name} finished for steamid {steamid}")
    return (steamid, results)

def debug_calculate_unfinished_precision_and_recall(folder_name):
    # first we need to get the results
    results_list = []
    test_data = pd.read_csv("data/player_games_test.csv")
    for file in os.listdir(os.path.join(result_path, folder_name)):
        if file.endswith("_results.csv"):
            steamid = int(file.split("_")[0])
            results = pd.read_csv(os.path.join(result_path, folder_name, file))
            results_list.append((steamid, results))
    calculate_precision_and_recall(folder_name, results_list, test_data)
    return results_list


def calculate_precision_and_recall(recommender_name, results_list, test_data):
    print(f"Calculating precision and recall for {recommender_name}...")
    # now we can calculate precision and recall
    precision = []
    precision_at_5 = []
    precision_at_10 = []
    precision_at_12 = []  # steam only shows 12 games
    precision_at_20 = []
    recall = []
    recall_at_5 = []
    recall_at_10 = []
    recall_at_12 = []  # steam only shows 12 games
    recall_at_20 = []
    
    for steamid, results in results_list:
        # first we need to get the test data for this user
        test_data_games = set(test_data[test_data["steamid"] == steamid]["appid"])

        # now we can calculate precision and recall
        if isinstance(results, pd.DataFrame):
            if "appid" not in results.columns:
                results = results.index.values  # assume the index is the appid
                if isinstance(results, np.int64):
                    results = [results]
            else:
                results = results["appid"].values
            if len(results) == 0:
                print(f"User {steamid} has no recommendations for {recommender_name}, might want to check")
                precision.append(0)
                precision_at_5.append(0)
                precision_at_10.append(0)
                precision_at_12.append(0)
                precision_at_20.append(0)
                recall.append(0)
                recall_at_5.append(0)
                recall_at_10.append(0)
                recall_at_12.append(0)
                recall_at_20.append(0)
                continue
            precision.append(len(set(results).intersection(test_data_games)) / len(results))
            precision_at_5.append(len(set(results[:5]).intersection(test_data_games)) / 5)
            precision_at_10.append(len(set(results[:10]).intersection(test_data_games)) / 10)
            precision_at_12.append(len(set(results[:12]).intersection(test_data_games)) / 12)
            precision_at_20.append(len(set(results[:20]).intersection(test_data_games)) / 20)
            recall.append(len(set(results).intersection(test_data_games)) / len(test_data_games))
            recall_at_5.append(len(set(results[:5]).intersection(test_data_games)) / len(test_data_games))
            recall_at_10.append(len(set(results[:10]).intersection(test_data_games)) / len(test_data_games))
            recall_at_12.append(len(set(results[:12]).intersection(test_data_games)) / len(test_data_games))
            recall_at_20.append(len(set(results[:20]).intersection(test_data_games)) / len(test_data_games))
        else:
            raise ValueError(f"Results for {recommender_name} and {steamid} is not a DataFrame")
        
    # now we can store the results
    # first we need to create a folder for the results of this recommender system
    if not os.path.exists(os.path.join(result_path, recommender_name)):
        os.mkdir(os.path.join(result_path, recommender_name))
    # now we can store the results, 
    pd.DataFrame({
        "steamid": [res[0] for res in results_list],
        "precision": precision,
        "precision_at_5": precision_at_5,
        "precision_at_10": precision_at_10,
        "precision_at_12": precision_at_12,
        "precision_at_20": precision_at_20,
        "recall": recall,
        "recall_at_5": recall_at_5,
        "recall_at_10": recall_at_10,
        "recall_at_12": recall_at_12,
        "recall_at_20": recall_at_20
    }).to_csv(os.path.join(result_path, recommender_name, "results.csv"), index=False)

def recommender_logic(recommender_system: recommender.AbstractRecommenderSystem, recommender_name: str, steamids: list, test_data: pd.DataFrame):
    # first we need to get the name of the recommender system
    start_time = time.time()
    # before we run the recommender system, check if there's already results
    if os.path.exists(os.path.join(result_path, recommender_name, "results.csv")):
        print(f"Recommender system {recommender_name} already ran, skipping...")
        return
    print(f"Running recommender system {recommender_name}...")
    # now we need to create a folder for the results of this recommender system
    if not os.path.exists(os.path.join(result_path, recommender_name)):
        os.mkdir(os.path.join(result_path, recommender_name))
    # now we can run the recommender system, saving results to calculate precision and recall later
    results_list = []
    if isinstance(recommender_system, recommender.AbstractRecommenderSystem):
        if paralellize_recommendations_mode == PROCESS_MODE:
            futures = [process_executor.submit(recommend_user, recommender_system, recommender_name, steamid) for steamid in steamids]
            for future in cf.as_completed(futures):
                results_list.append(future.result())
        elif paralellize_recommendations_mode == THREAD_MODE:
            futures = [thread_executor.submit(recommend_user, recommender_system, recommender_name, steamid) for steamid in steamids]
            for future in cf.as_completed(futures):
                results_list.append(future.result())
        elif paralellize_recommendations_mode == None:
            for steamid in steamids:
                results_list.append(recommend_user(recommender_system, recommender_name, steamid))
    else:
        raise ValueError("Recommender system is not an instance of AbstractRecommenderSystem")
    finish_time = time.time() - start_time
    with open(os.path.join(result_path, recommender_name, "time_in_seconds.txt"), "w") as f:
        f.write(str(finish_time))
    # now we can calculate precision and recall, comparing against the test data
    calculate_precision_and_recall(recommender_name, results_list, test_data)
    print(f"Recommender system {recommender_name} finished")
    del results_list
    del recommender_system  # TODO unsure if this works
    garbage_collector.collect()

process_executor = cf.ProcessPoolExecutor(6)
thread_executor = cf.ThreadPoolExecutor()

def handler(signum, frame):
    # print("Cancelling everything...")
    sys.exit(0)

# signal.signal(signal.SIGINT, handler)


paralellize_data_loading = False
if paralellize_data_loading:
    print("Are you sure you want to paralellize data loading? It's very RAM intensive and render your computer unusable.")
    print("Also, the classes have been optimized to use a single-threaded GLOBAL_CACHE which is much faster but not thread safe.")
    print("If you want to paralellize data loading, you need to change the code to use a thread-safe cache.")
    y_n = input("Do you still want to paralellize data loading? [y/N] ")
    if y_n.lower() != "y":
        paralellize_data_loading = False
parallelize_similarities_and_recommenders = False
PROCESS_MODE = 1
THREAD_MODE = 2
paralellize_recommendations_mode = None # PROCESS_MODE # THREAD_MODE

player_games_playtimes = []  # for python -i experiments.py
recommender_combinations = None # for python -i experiments.py
game_categories = []
game_genres = []
game_tags = []

def run_recommender_experiments(cull: int, interactive: bool, only_playtime: bool, nitpicked_steamids: bool):
    global recommender_combinations
    global player_games_playtimes
    # first check if the data/ folder exists
    if not os.path.exists("data"):
        os.mkdir("data")

    # now check if the data is already there
    if not os.path.exists("data/player_games_train.csv") or not os.path.exists("data/player_games_test.csv"):
        print("Data is not split, splitting now...")
        split_train_test.main("data/player_games.csv")
    # now load the data
    print("Loading data...")
    train_data = pd.read_csv("data/player_games_train.csv")
    test_data = pd.read_csv("data/player_games_test.csv")
    print("Data loaded. Loading normalization classes and similarity classes...")
    # now get all the normalization classes
    normalization_classes = [cls for cls in normalization.__dict__.values() if isinstance(cls, type) and issubclass(cls, normalization.AbstractPlaytimeNormalizer) and cls != normalization.AbstractPlaytimeNormalizer and cls != normalization.NoNormalization]
    # now get all the similarities, separated by game_similarities and user_similarities
    # similarities = [sim for sim in recommender.__dict__.values() if isinstance(sim, type) and issubclass(sim, recommender.AbstractSimilarity)]
    game_similarity_types = [sim for sim in recommender.__dict__.values() if isinstance(sim, type) and issubclass(sim, recommender.AbstractGameSimilarity) and sim != recommender.AbstractGameSimilarity and sim != recommender.GameDetailsSimilarity and not issubclass(sim, recommender.RawGameTagSimilarity)]
    game_tag_similarity_types = [sim for sim in recommender.__dict__.values() if isinstance(sim, type) and issubclass(sim, recommender.RawGameTagSimilarity)]
    
    user_similarity_types = [sim for sim in recommender.__dict__.values() if isinstance(sim, type) and issubclass(sim, recommender.RawUserSimilarity)]
    print("Normalization classes and similarity classes loaded.\nInstantiating player game data with different playtime normalizers. This might take very long...")
    # now we want to mix recommender.PlayerGamesPlaytime with every normalization class with thresholds from 0.1 to 0.9
    # first we need to get all the combinations of normalization classes and thresholds
    # to instantiate every PlayerGamesPlaytime with every normalization class and threshold
    
    player_games_minhash_thresholds = [0.6, 0.75]
    pg_relevant_thresholds = [0.6, 0.8]
    print("Instantiating PlayerGamesPlaytimes with different thresholds and normalizers in serial...")
    for normalization_class in normalization_classes:
        for minhash_threshold in player_games_minhash_thresholds:
            for relevant_threshold in pg_relevant_thresholds:
                player_games_playtimes.append(recommender.PlayerGamesPlaytime(train_data, normalization_class(), relevant_threshold, minhash_threshold))
    if interactive:
        input("If running with python -i experiments.py, you can now access player_games_playtimes. Press CTRL-C to start interactive shell or Enter to continue...")
    # now we want to mix recommender.PlayerGamesPlaytime with every recommender class
    # we can handwrite part of this since each recommender takes different arguments
    # first we need to get all the combinations of recommender classes and thresholds
    # but first we need to get different game_similarities and user_similarities
    # and to get different game and user similarities, we'd first need every AbstractRecommenderData except for PlayerGamesPlaytime and AbstractRecommenderData
    # also doing this programatically is harder than doing it by hand, I just took this from recommender_shell.py
    if not only_playtime:
        print("Instantiating basic Recommender Data classes...")
        game_similarity_thresholds = np.append(np.linspace(0.3, 0.8, 5), [1, 1.01])  # threshold 1 means linear search / other methods will be used
        # game_details = recommender.GameDetails('data/game_details.csv')  # this one is global
        game_details = None
        game_developers = recommender.GameDevelopers('data/game_developers_empty.csv')
        game_publishers = recommender.GamePublishers('data/game_publishers_empty.csv')
        game_categories_csv = pd.read_csv('data/game_categories_empty.csv')
        game_genres_csv = pd.read_csv('data/game_genres_empty.csv')
        game_tags_csv = pd.read_csv('data/game_tags_empty.csv')
        idf_weights = np.linspace(0, 0.6, 3)
        weight_thresholds = [0.75, 1] # np.linspace(0.5, 1, 3)
        
        print("Instantiating complex Recommender Data with different thresholds in serial...")
        for minhash_threshold in game_similarity_thresholds:
            game_categories.append(recommender.GameCategories(game_categories_csv, minhash_threshold))
            game_genres.append(recommender.GameGenres(game_genres_csv, minhash_threshold))
            if minhash_threshold > 1.0:
                for weight_threshold in weight_thresholds:
                    for idf_weight in idf_weights:
                        game_tags.append(recommender.GameTags(game_tags_csv, weight_threshold, minhash_threshold, idf_weight))
            else:
                for idf_weight in idf_weights:
                    game_tags.append(recommender.GameTags(game_tags_csv, 1, minhash_threshold, idf_weight))
        if interactive:
            input("If running with python -i experiments.py, you can now access game_categories, game_genres, and game_tags. Press CTRL-C to start interactive shell or Enter to continue...")

        game_info = {}
        # game_tags_dict = { (threshold, weight_threshold, idf_weight) : recommender.GameTags(game_tags_csv, weight_threshold, threshold, idf_weight) for threshold in game_similarity_thresholds for weight_threshold in weight_thresholds for idf_weight in idf_weights }
        for thres in game_similarity_thresholds:
            # find the index of each recommender data in each list of recommender datas where their threshold is equal to thres
            # then use that index to get the recommender data from each list
            game_category = None
            for gc in game_categories:
                if gc.threshold == thres:
                    game_category = gc
                    break
            game_genre = None
            for gg in game_genres:
                if gg.threshold == thres:
                    game_genre = gg
                    break
            
            game_info[thres] = recommender.GameInfo(game_details, game_category, game_developers, game_publishers, game_genre, None)
    # NOTE: this one and the next ones are probably faster in single-threaded mode

    if not only_playtime:
        print("Instantiating GameSimilarities and UserSimilarities with their respective recommender data in serial...")
        game_similarities = [] # [recommender.GameDetailsSimilarity(game_details)]
        for game_similarity_type in game_similarity_types:
            for thres in game_similarity_thresholds:
                game_similarities.append(game_similarity_type(game_info[thres]))
        for game_tag_similarity in game_tag_similarity_types:
            for game_tag in game_tags:
                game_similarities.append(game_tag_similarity(game_tag))
    print("Instantiating UserSimilarities with different thresholds and normalizers in serial...")
    user_similarities = []
    for user_similarity_type in user_similarity_types:
        for pgdata in player_games_playtimes:
            user_similarities.append(user_similarity_type(pgdata))

    recommender_combinations = [] # [recommender.RandomRecommenderSystem(), recommender.RatingBasedRecommenderSystem(game_details, player_games_playtimes[0])]
    if not only_playtime:
        print("Instantiating ContentBasedRecommenderSystem with different thresholds and normalizers in serial...")
        for game_similarity in game_similarities:
            recommender_combinations.append(recommender.ContentBasedRecommenderSystem(player_games_playtimes[0], game_similarity, 0.2, False))
    
    print("Instantiating PlaytimeBasedRecommenderSystem with different thresholds and normalizers in serial...")
    for player_games_playtime in player_games_playtimes:
        for user_similarity in user_similarities:
            recommender_combinations.append(recommender.PlaytimeBasedRecommenderSystem(player_games_playtime, user_similarity))
    # now we have all the combinations, we can run them
    # first we need to create a folder to store the results
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    # but get the steamids first
    if nitpicked_steamids:
        import pickle
        steamids = pickle.load(open("data/nitpicked_steamids.pickle", "rb"))
    else:
        steamids = test_data["steamid"].unique()
        if cull > 0:
            steamids = steamids[:cull]
    # now we can run the experiments
    for recommender_system in recommender_combinations:
        recommender_name = repr(recommender_system)
        recommender_logic(recommender_system, recommender_name, steamids, test_data)
    
    print("Finished running all recommender systems. Calculating average precision and recall, and plotting...")

results_for_rec = {}
def plot_results():
    global final_results, results_for_rec
    # now we can calculate the average precision and recall for each recommender system
    # first we need to get the names of the recommender systems
    recommender_names = [ f.path for f in os.scandir(result_path) if f.is_dir() ]
    recommender_names = [recommender_name.split("/")[-1] for recommender_name in recommender_names]
    # now we can calculate the average precision and recall
    average_precision = []
    average_precision_at_5 = []
    average_precision_at_10 = []
    average_precision_at_12 = []
    average_precision_at_20 = []
    average_recall = []
    average_recall_at_5 = []
    average_recall_at_10 = []
    average_recall_at_12 = []
    average_recall_at_20 = []
    for recommender_name in recommender_names:
        # first we need to get the results
        try:
            results = pd.read_csv(os.path.join(result_path, recommender_name, "results.csv"))
            results_for_rec[recommender_name] = results
            # now we can calculate the average precision and recall
            average_precision.append(results["precision"].mean())
            average_precision_at_5.append(results["precision_at_5"].mean())
            average_precision_at_10.append(results["precision_at_10"].mean())
            average_precision_at_12.append(results["precision_at_12"].mean())
            average_precision_at_20.append(results["precision_at_20"].mean())
            average_recall.append(results["recall"].mean())
            average_recall_at_5.append(results["recall_at_5"].mean())
            average_recall_at_10.append(results["recall_at_10"].mean())
            average_recall_at_12.append(results["recall_at_12"].mean())
            average_recall_at_20.append(results["recall_at_20"].mean())
        except FileNotFoundError:
            print("Could not find results for recommender system (maybe it hasn't finished?): " + recommender_name)
            average_precision.append(np.NaN)
            average_precision_at_5.append(np.NaN)
            average_precision_at_10.append(np.NaN)
            average_precision_at_12.append(np.NaN)
            average_precision_at_20.append(np.NaN)
            average_recall.append(np.NaN)
            average_recall_at_5.append(np.NaN)
            average_recall_at_10.append(np.NaN)
            average_recall_at_12.append(np.NaN)
            average_recall_at_20.append(np.NaN)

    # now we can store the results
    final_results = pd.DataFrame({
        "recommender_system": recommender_names,
        "average_precision": average_precision,
        "average_precision_at_5": average_precision_at_5,
        "average_precision_at_10": average_precision_at_10,
        "average_precision_at_12": average_precision_at_12,
        "average_precision_at_20": average_precision_at_20,
        "average_recall": average_recall,
        "average_recall_at_5": average_recall_at_5,
        "average_recall_at_10": average_recall_at_10,
        "average_recall_at_12": average_recall_at_12,
        "average_recall_at_20": average_recall_at_20
    })
    final_results.to_csv(os.path.join(result_path, "average_results.csv"), index=False)
    print("Final results saved to results/average_results.csv. Plotting and saving to results/results.png...")
    # now we can plot the results
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].plot(final_results["recommender_system"], final_results["average_precision_at_5"], label="precision@5")
    ax[0].plot(final_results["recommender_system"], final_results["average_precision_at_10"], label="precision@10")
    ax[0].plot(final_results["recommender_system"], final_results["average_precision_at_12"], label="precision@12")
    ax[0].plot(final_results["recommender_system"], final_results["average_precision_at_20"], label="precision@20")
    ax[0].set_title("Precision")
    ax[0].set_xlabel("Recommender system")
    ax[0].set_ylabel("Precision")
    ax[0].legend()
    ax[1].plot(final_results["recommender_system"], final_results["average_recall_at_5"], label="recall@5")
    ax[1].plot(final_results["recommender_system"], final_results["average_recall_at_10"], label="recall@10")
    ax[1].plot(final_results["recommender_system"], final_results["average_recall_at_12"], label="recall@12")
    ax[1].plot(final_results["recommender_system"], final_results["average_recall_at_20"], label="recall@20")
    ax[1].set_title("Recall")
    ax[1].set_xlabel("Recommender system")
    ax[1].set_ylabel("Recall")
    ax[1].legend()
    fig.savefig(os.path.join(result_path, "results.png"))
    # fig.show()

topN_steamid = []
def find_topN_from_results(res_dict, N=200, K=2.5):
    # debugging purposes and to take the "top" users
    global topN_steamid
    topkN_per_rec = {}
    kN = int(np.floor(K*N))
    for recommender_name, results in res_dict.items():
        # order the results by the precision
        results = results.sort_values(by="precision", ascending=False)
        # get the top N*K
        res_kn = results.head(kN)
        # first check if any's precision is 0
        if res_kn["precision"].min() == 0:
            print(f"WARNING: Recommender system {recommender_name} has a precision of 0 in the top {kN} results. Filtering out those users...")
            res_kn = res_kn[res_kn["precision"] > 0]
        more_than_0 = res_kn["steamid"].tolist()
        if len(more_than_0) < N:
            print(f"WARNING: Recommender system {recommender_name} has {len(more_than_0)} results, less than N={N}. Skipping this recommender system...")
            continue
        topkN_per_rec[recommender_name] = more_than_0
        print(f"Found {len(more_than_0)} users with more than 0 precission for recommender system {recommender_name}.")
    # now we can find the top N
    topN_steamid = list(set.intersection(*map(set, topkN_per_rec.values())))
    if len(topN_steamid) < N:
        print(f"WARNING: Found less than {N} 'colliding' users... Edit the code to find more users.")
    elif len(topN_steamid) > N:
        print(f"GOOD SIGN: Found more than {N} 'colliding' users... Filtering based on how many times they appear in the top {kN} of the recommender systems.")
        # now we need to filter the topN_steamid list
        # we will count how many times each user appears in the top kN of the recommender systems
        # and then we will filter the topN_steamid list based on that
        # we will use a dict to count the number of times each user appears in any topkN_per_rec
        user_count = {}
        for recommender_name, topkN in topkN_per_rec.items():
            for user in topkN:
                if user in user_count:
                    user_count[user] += 1
                else:
                    user_count[user] = 1
        # now we can filter the topN_steamid list
        sorted_user_count = sorted(user_count.items(), key=lambda x: x[1], reverse=True)
        topN_steamid = [user[0] for user in sorted_user_count[:N]]
    print(f"Found top {len(topN_steamid)} users from the results of the recommender systems. Stored in topN_steamid.")
    

if __name__ == "__main__":
    try:
        # argparse to skip to results
        parser = argparse.ArgumentParser()
        parser.add_argument("--skip_rec", action="store_true", help="Skip running the recommender systems and go straight to the results")
        # also to skip plotting results
        parser.add_argument("--skip_plot", action="store_true", help="Skip plotting the results")
        parser.add_argument("--cull", type=int, default=-1, help="Cull the users to test. Mostly debugging purposes.")
        parser.add_argument("-pt","--only_playtime", action="store_true", help="Only compute playtime, useful when done with the Content Based recommender systems.")
        parser.add_argument("-np","--use_nitpicked_steamids", action="store_true", help="Use the nitpicked steamids instead of the topN_steamid list. Useful for debugging purposes.")
        # parser.add_argument("--interactive", action="store_true", help="Run the program in interactive mode. Mostly for debugging purposes.")
        parser.add_argument("-i", "--interactive", action="store_true", help="Run the program in interactive mode. Mostly for debugging purposes.")
        args = parser.parse_args()
        if args.skip_rec:
            print("Skipping running recommender systems...")
        else:
            run_recommender_experiments(args.cull, args.interactive, args.only_playtime, args.use_nitpicked_steamids)
        if args.skip_plot:
            print("Skipping plotting results...")
        else:
            plot_results()
            # used for debugging purposes
            if args.interactive:
                # note: if we tried to get them from anything other than 'GameTag' recommender similarities,
                # we'd have 7 users with a precission higher than 0 in all recommender systems
                # we're using this one to find potential users to test with Playtime recommender systems, 
                # and we want to have a subset that doesn't yield 0 results for some users after hours of computation
                game_tag_res = {key: item for key, item in results_for_rec.items() if "GameTag" in key}
                find_topN_from_results(game_tag_res, N=100, K=4)
    except KeyboardInterrupt:
        print("**************\nKeyboardInterrupt detected. Stopping executor...\n**************")
        process_executor.shutdown(wait=False, cancel_futures=True)
        print("Executor stopped. Exiting gracefully...")
        sys.exit(0)
