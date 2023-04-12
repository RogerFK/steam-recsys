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

# Our goal is to explore which thresholds, mix of normalization methods and recommenders/mix of recommenders work best.
# This script assumes you have already run the split_train_test.py script and your data is inside the data/ folder.
# If the data isn't there, we will split the data and save it to the data/ folder using the default values.
import split_train_test

def recommend_user(recommender_system: recommender.AbstractRecommenderSystem, recommender_name, steamid: int):
    # beforehand, check if there's results already
    results_file = os.path.join("results", recommender_name, f"{steamid}_results.csv")
    if os.path.exists(results_file):
        return (steamid, pd.read_csv(results_file))
    results = recommender_system.recommend(steamid, n=50, filter_owned=True)
    # results_list.append((steamid, results))
    # store the results, which are a DataFrame (or if they're a list of tuples, we can convert it to a DataFrame)
    if not os.path.exists(os.path.join("results", recommender_name)):
        os.mkdir(os.path.join("results", recommender_name))
    if isinstance(results, list):
        results = pd.DataFrame(results, columns=["appid", "score"])
    results.to_csv(os.path.join("results", recommender_name, f"{steamid}_results.csv"), index=True)
    print(f"Recommender system {recommender_name} finished for steamid {steamid}")
    return (steamid, results)

process_executor = cf.ProcessPoolExecutor(6)
threading_executor = cf.ThreadPoolExecutor()

def handler(signum, frame):
    # print("Cancelling everything...")
    sys.exit(0)

# signal.signal(signal.SIGINT, handler)

# change BIN_DATA_PATH to "bin_data_test" to avoid conflicts
recommender.BIN_DATA_PATH = "bin_data_test"
paralellize_data_loading = False
paralellize_everything = False

def main():
    # first check if the data/ folder exists
    if not os.path.exists("data"):
        os.mkdir("data")

    # now check if the data is already there
    if not os.path.exists("data/player_games_train.csv") or not os.path.exists("data/player_games_test.csv"):
        print("Data is not split, splitting now...")
        split_train_test.main("data/player_games_subset.csv")
    # now load the data
    print("Loading data...")
    train_data = pd.read_csv("data/player_games_train.csv")
    test_data = pd.read_csv("data/player_games_test.csv")
    print("Data loaded. Loading normalization classes and similarity classes...")
    # now get all the normalization classes
    normalization_classes = [cls for cls in normalization.__dict__.values() if isinstance(cls, type) and issubclass(cls, normalization.AbstractPlaytimeNormalizer) and cls != normalization.AbstractPlaytimeNormalizer]
    normalization_classes.append(normalization.RootPlaytimeNormalizer)  # we add another one as
    # now get all the similarities, separated by game_similarities and user_similarities
    # similarities = [sim for sim in recommender.__dict__.values() if isinstance(sim, type) and issubclass(sim, recommender.AbstractSimilarity)]
    game_similarity_types = [sim for sim in recommender.__dict__.values() if isinstance(sim, type) and issubclass(sim, recommender.AbstractGameSimilarity) and sim != recommender.AbstractGameSimilarity and sim != recommender.GameDetailsSimilarity]
    user_similarity_types = [sim for sim in recommender.__dict__.values() if isinstance(sim, type) and issubclass(sim, recommender.RawUserSimilarity)]
    print("Normalization classes and similarity classes loaded.\nInstantiating player game data with different playtime normalizers. This might take very long...")
    # now we want to mix recommender.PlayerGamesPlaytime with every normalization class with thresholds from 0.1 to 0.9
    # first we need to get all the combinations of normalization classes and thresholds
    # to instantiate every PlayerGamesPlaytime with every normalization class and threshold
    player_games_playtimes = []
    player_games_thresholds = np.linspace(0.15, 0.75, 5)
    if paralellize_data_loading:
        print("Instantiating PlayerGamesPlaytimes with different thresholds and normalizers in parallel...")
        futures = [process_executor.submit(recommender.PlayerGamesPlaytime, train_data, normalization_class(), threshold) for normalization_class in normalization_classes for threshold in player_games_thresholds ] 
        for future in cf.as_completed(futures):
            player_games_playtimes.append(future.result())
    else:
        # NOTE: This one is WAY faster and wastes less RAM (which was somewhat expected)
        # TODO: Cache PlayerGamesPlaytime objects to avoid reloading them with some global dict if they've been loaded
        print("Instantiating PlayerGamesPlaytimes with different thresholds and normalizers in serial...")
        for normalization_class in normalization_classes:
            for threshold in player_games_thresholds:
                player_games_playtimes.append(recommender.PlayerGamesPlaytime(train_data, normalization_class(), threshold))
    
    input("Che vamos bien de RAM?")
    # now we want to mix recommender.PlayerGamesPlaytime with every recommender class
    # we can handwrite part of this since each recommender takes different arguments
    # first we need to get all the combinations of recommender classes and thresholds
    # but first we need to get different game_similarities and user_similarities
    # and to get different game and user similarities, we'd first need every AbstractRecommenderData except for PlayerGamesPlaytime and AbstractRecommenderData
    # also doing this programatically is harder than doing it by hand, I just took this from recommender_shell.py
    
    print("Instantiating Recommender Data with different thresholds in parallel...")
    game_similarity_thresholds = np.linspace(0.3, 0.8, 5)
    game_details = recommender.GameDetails('data/game_details.csv')  # this one is global
    game_developers = recommender.GameDevelopers('data/game_developers.csv')
    game_publishers = recommender.GamePublishers('data/game_publishers.csv')
    game_categories_csv = pd.read_csv('data/game_categories.csv')
    game_categories_futures = [process_executor.submit(recommender.GameCategories, game_categories_csv, threshold) for threshold in game_similarity_thresholds ]
    game_genres_csv = pd.read_csv('data/game_genres.csv')
    game_genres_futures = [process_executor.submit(recommender.GameGenres, game_genres_csv, threshold) for threshold in game_similarity_thresholds ]
    game_tags_csv = pd.read_csv('data/game_tags.csv')
    idf_weights = np.linspace(0, 0.3, 3)
    weight_thresholds = np.linspace(0.5, 1, 3)
    game_tags_futures = [process_executor.submit(recommender.GameTags, game_tags_csv, weight_threshold, threshold, idf_weight) for threshold in game_similarity_thresholds for weight_threshold in weight_thresholds for idf_weight in idf_weights ]
    game_categories = []
    game_genres = []
    game_tags = []
    for future in cf.as_completed(game_categories_futures + game_genres_futures + game_tags_futures):
        if future in game_categories_futures:
            game_categories.append(future.result())
        elif future in game_genres_futures:
            game_genres.append(future.result())
        elif future in game_tags_futures:
            game_tags.append(future.result())
        else:
            raise ValueError(f"Future {repr(future)} not in any list" +
                             f"game_categories_futures: {game_categories_futures}\n" + f"game_categories: {game_categories}\n\n"
                             f"game_genres_futures: {game_genres_futures}\n" + f"game_genres: {game_genres}\n\n"
                             f"game_tags_futures: {game_tags_futures}\n" + f"game_tags: {game_tags}\n\n"
                             )
    # game_categories = [future.result() for future in game_categories_futures]
    # game_developers = [future.result() for future in game_developers_futures]
    # game_publishers = [future.result() for future in game_publishers_futures]
    # game_genres = [future.result() for future in game_genres_futures]
    # game_tags = [future.result() for future in game_tags_futures]
    game_info = {}
    for thres in game_similarity_thresholds:
        # find the index of each recommender data in each list of recommender datas where their threshold is equal to thres
        # then use that index to get the recommender data from each list
        game_category = None
        for gc in game_categories:
            if gc.lshensemble.threshold == thres:
                game_category = gc
                break
        game_genre = None
        for gg in game_genres:
            if gg.lshensemble.threshold == thres:
                game_genre = gg
                break
        game_tag = None
        for gt in game_tags:
            if gt.lshensemble.threshold == thres:
                game_tag = gt
                break
        
        game_info[thres] = recommender.GameInfo(game_details, game_category, game_developers, game_publishers, game_genre, game_tag)
    # NOTE: this one and the next ones are probably faster in single-threaded mode
    if paralellize_everything:
        print("Instantiating GameSimilarities with their respective recommender data in parallel...")

        game_similarity_futures = [process_executor.submit(game_similarity_type, game_info[thres]) for game_similarity_type in game_similarity_types for thres in game_similarity_thresholds]
        game_similarities = [recommender.GameDetailsSimilarity(game_details)]
        for future in cf.as_completed(game_similarity_futures):
            game_similarities.append(future.result())
            
        print("Instantiating UserSimilarities with their respective recommender data in parallel...")
        user_similarity_futures = [process_executor.submit(user_similarity_type, pgdata) for user_similarity_type in user_similarity_types for pgdata in player_games_playtimes]
        user_similarities = []
        for future in cf.as_completed(user_similarity_futures):
            user_similarities.append(future.result())
    else:
        print("Instantiating GameSimilarities and UserSimilarities with their respective recommender data in serial...")
        game_similarities = [recommender.GameDetailsSimilarity(game_details)]
        for game_similarity_type in game_similarity_types:
            for thres in game_similarity_thresholds:
                game_similarities.append(game_similarity_type(game_info[thres]))
        user_similarities = []
        for user_similarity_type in user_similarity_types:
            for pgdata in player_games_playtimes:
                user_similarities.append(user_similarity_type(pgdata))

    recommender_combinations = [recommender.RandomRecommenderSystem(), recommender.RatingBasedRecommenderSystem(game_details, player_games_playtimes[0])]
    if paralellize_everything:
        print("Instantiating ContentBasedRecommenderSystem with different thresholds and normalizers in parallel...")
        futures = [process_executor.submit(recommender.ContentBasedRecommenderSystem, player_games_playtime, game_similarity) for player_games_playtime in player_games_playtimes for game_similarity in game_similarities]
        for future in cf.as_completed(futures):
            recommender_combinations.append(future.result())
        print("Instantiating PlaytimeBasedRecommenderSystem with different thresholds and normalizers in parallel...")
        futures = [process_executor.submit(recommender.PlaytimeBasedRecommenderSystem, player_games_playtime, user_similarity) for player_games_playtime in player_games_playtimes for user_similarity in user_similarities]
        for future in cf.as_completed(futures):
            recommender_combinations.append(future.result())
    else:
        print("Instantiating ContentBasedRecommenderSystem with different thresholds and normalizers in serial...")
        for player_games_playtime in player_games_playtimes:
            for game_similarity in game_similarities:
                recommender_combinations.append(recommender.ContentBasedRecommenderSystem(player_games_playtime, game_similarity))
        print("Instantiating PlaytimeBasedRecommenderSystem with different thresholds and normalizers in serial...")
        for player_games_playtime in player_games_playtimes:
            for user_similarity in user_similarities:
                recommender_combinations.append(recommender.PlaytimeBasedRecommenderSystem(player_games_playtime, user_similarity))
    # now we have all the combinations, we can run them
    # first we need to create a folder to store the results
    if not os.path.exists("results"):
        os.mkdir("results")
    # but get the steamids first
    steamids = test_data["steamid"].unique()[:12]  # TODO change this to [:12] when testing
    # now we can run the experiments
    for recommender_system in recommender_combinations:
        # first we need to get the name of the recommender system
        recommender_name = repr(recommender_system)
        start_time = time.time()
        # before we run the recommender system, check if there's already results
        if os.path.exists(os.path.join("results", recommender_name, "results.csv")):
            print(f"Recommender system {recommender_name} already ran, skipping...")
            continue
        print(f"Running recommender system {recommender_name}...")
        # now we need to create a folder for the results of this recommender system
        if not os.path.exists(os.path.join("results", recommender_name)):
            os.mkdir(os.path.join("results", recommender_name))
        # now we can run the recommender system, saving results to calculate precision and recall later
        results_list = []
        if isinstance(recommender_system, recommender.AbstractRecommenderSystem):
            futures = [process_executor.submit(recommend_user, recommender_system, recommender_name, steamid) for steamid in steamids]
            for future in cf.as_completed(futures):
                results_list.append(future.result())
        else:
            raise ValueError("Recommender system is not an instance of AbstractRecommenderSystem")
        finish_time = time.time() - start_time
        with open(os.path.join("results", recommender_name, "time_in_seconds.txt"), "w") as f:
            f.write(str(finish_time))
        # now we can calculate precision and recall, comparing against the test data
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
        if not os.path.exists(os.path.join("results", recommender_name)):
            os.mkdir(os.path.join("results", recommender_name))
        # now we can store the results, 
        pd.DataFrame({
            "steamid": steamids,
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
        }).to_csv(os.path.join("results", recommender_name, "results.csv"), index=False)
        print(f"Recommender system {recommender_name} finished")
    
    print("Finished running all recommender systems. Calculating average precision and recall, and plotting...")
    # now we can calculate the average precision and recall for each recommender system
    # first we need to get the names of the recommender systems
    recommender_names = [repr(recommender_system) for recommender_system in recommender_combinations]
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
        results = pd.read_csv(os.path.join("results", recommender_name, "results.csv"))
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
    final_results.to_csv(os.path.join("results", "average_results.csv"), index=False)
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
    fig.savefig(os.path.join("results", "results.png"))
    fig.show()

def plot_results():
    # TODO: plot the results, read the results from the results folder
    pass

if __name__ == "__main__":
    try:
        # argparse to skip to results
        parser = argparse.ArgumentParser()
        parser.add_argument("--skip_rec", action="store_true", help="Skip running the recommender systems and go straight to the results")
        # also to skip plotting results
        parser.add_argument("--skip_plot", action="store_true", help="Skip plotting the results")
        args = parser.parse_args()
        if args.skip_rec:
            print("Skipping running recommender systems...")
        else:
            main()
        if args.skip_plot:
            print("Skipping plotting results...")
        else:
            plot_results()
    except KeyboardInterrupt:
        print("**************\nKeyboardInterrupt detected. Stopping executor...\n**************")
        process_executor.shutdown(wait=False, cancel_futures=True)
        print("Executor stopped. Exiting gracefully...")
        sys.exit(0)
