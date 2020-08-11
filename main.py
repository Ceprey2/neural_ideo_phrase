import csv
import json

import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, request
from flask import render_template
from scipy.cluster import hierarchy as sch
from scipy.cluster.hierarchy import fcluster, leaders, centroid
from scipy.cluster.hierarchy import linkage, dendrogram, ward, cut_tree
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

number_of_clusters = 3



def print_cut_tree_features(cut_tr, names):

    for element in cut_tr:
        print(element[0], names[element[0]])

    pass


def get_centroid_feature_for_cluster(descriptors):
    # print("descriptors from getting centroids")
    # print(descriptors)
    list_descriptors = []

    for descrs in descriptors:
       list_descriptors.extend(descrs.split())

    all_normalized_descriptors = [ds.replace("*", "").strip().lower() for ds in list_descriptors]

    if (len (all_normalized_descriptors)) > 0:

        return max(set(all_normalized_descriptors), key=all_normalized_descriptors.count) # searching for mode

    else:
        return 'empty_center'




def transform_labels_to_names(labels_ward_1, labels_ward_2, descriptors):
    labels_ward_1_named = []
    labels_ward_2_named = []

    # print(get_centroid_feature_for_cluster([descrs for descrs in descriptors[labels_ward_2 ==2] ]))

    for lbl in labels_ward_1:
        # print("label")
        # print(lbl)
        labels_ward_1_named.append(
            get_centroid_feature_for_cluster([descrs for descrs in descriptors[labels_ward_1 == lbl]]))

    for lbl in labels_ward_2:
        # print("label")
        # print(lbl)
        labels_ward_2_named.append(
            get_centroid_feature_for_cluster([descrs for descrs in descriptors[labels_ward_2 == lbl]]))

    return  labels_ward_1_named, labels_ward_2_named


def hierarchical_clustering(dict_from_csv,current_langugage):
    dict_subclusters_array_hierarchical = []

    wcss = []
    entries = []


    df_dict_from_csv = pd.DataFrame(dict_from_csv)
    descriptors = df_dict_from_csv[current_langugage+'hetmans']
    phrases =  df_dict_from_csv[current_langugage]
    df_dict_from_csv.fillna('null', inplace=True)

    # Calculate the linkage: mergings
    tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+'?\w+\b")
    X = tfidf.fit_transform(descriptors).todense()
    mergings = linkage(X, method="complete")

    Z_ward = ward(X)


   # ClusterNode(Z[1])
    names = tfidf.get_feature_names()
    print("len(names)")
    print(len(names))

    cutree = cut_tree(mergings,  height=1.2)
    #cutree = cut_tree(Z_ward,  n_clusters=24)

    # print("cutree")
    # print(cutree)
    # print(cutree.shape)

    print ("height 1.2")
    #print_cut_tree_features(cutree, names)

    cutree = cut_tree(mergings, height=1.3)
    #print ("height 1.3")
    #print_cut_tree_features(cutree, names)

    cutree = cut_tree(mergings, height=1)
    #print("height 1")
    #print_cut_tree_features(cutree, names)

    cutree = cut_tree(mergings, height=0)
    #print("height 0")
    #print_cut_tree_features(cutree, names)


    to_tree_obj = sch.to_tree(Z_ward)
    #print("to_tree_obj")
    #print(to_tree_obj)
    #print("pre_order length")
    #print(len(to_tree_obj.pre_order()))
    #print("to_tree_obj.get_count()")
    #print(to_tree_obj.get_count())

    #Y = ward(pdist(X))
    #Z = linkage(Y)
    #leaders = scipy.cluster.hierarchy.leaders(X, Z)



    y_dist = pdist(X)
    Z_centroid = centroid(y_dist)


    print("leaders")
    print(leaders)
    dendrogram(mergings,

               leaf_rotation=90,
               leaf_font_size=6,
               )
    plt.show()

    labels = sch.fcluster(mergings, 1, criterion='distance')
    # print("mergings t=10 labels", labels)
    #
    # print("mergings shape")
    # print(mergings.shape)
    # print(mergings[0])
    # print(mergings[2])
    # print(mergings[248])
    # print(mergings[286])
    #
    # print(mergings[0][0])
    # print("Z_ward shape")
    # print(Z_ward.shape)
    # print("centroid shape")
    # print(Z_centroid.shape)

    # idxs = [33, 68, 62]
    # plt.figure(figsize=(10, 8))
    #
    # plt.scatter([X[:,0]],[X[:,1]])# plot all points
    # plt.scatter([X[idxs, 0]], [X[idxs, 1]], c='r')  # plot interesting points in red again
    # plt.show()

    cutree = cut_tree(Z_centroid, height=1)
    #print("height 1")
    #print_cut_tree_features(cutree, names)

    cutree = cut_tree(Z_centroid, height=3)
    #print("height 3")
    #print_cut_tree_features(cutree, names)

    labeled_rows_by_numbers_ward_level1 = fcluster(Z_ward, 2,
                             criterion='distance')  # LET'S ASSIGN NUMBER ONE TO THE UPPER LEVEL. HEIGT 2 IS HIGHER THAN 1 IN THE TREE


    labeled_rows_by_number_ward_level2 = fcluster(Z_ward, 1, criterion='distance')

    labels_ward_1_named, labels_ward_2_named = transform_labels_to_names(labeled_rows_by_numbers_ward_level1, labeled_rows_by_number_ward_level2, descriptors)

    pd.set_option('display.max_rows', None)  # TO AVOID TRUNCATING TABLE WHEN PRINTING
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 1000)
    df_ward = pd.DataFrame({'phrases': phrases, 'labels2': labels_ward_2_named, 'labels1': labels_ward_1_named})


    print("DataFrame of two level labels")
    print(df_ward)

    for lb1 in set(labels_ward_1_named):
        entries = []
        print("label_number")
        print(lb1)

        dict_centroid_phrases_hierarchical_subcluster = {"main_centroid": lb1,
                                                         "entries": entries}

        for row_number in range(len(dict_from_csv)):


            if ((lb1 == labels_ward_1_named[row_number])):
             entry = {
                        "centroid": labels_ward_2_named[row_number],

                        "ukrlangphrases": df_dict_from_csv["ukrlang"][row_number],
                        "spanlangphrases": df_dict_from_csv["spanishlang"][row_number],
                        "engllangphrases": df_dict_from_csv["engllang"][row_number],
                        "frenchlangphrases": df_dict_from_csv["frenchlang"][row_number],
                        "itallangphrases": df_dict_from_csv["itallang"][row_number],
                        "latinlangphrases": df_dict_from_csv["latinlang"][row_number],
                        "ruslangphrases": df_dict_from_csv["ruslang"][row_number],
                        "hebrlangphrases": df_dict_from_csv["hebrlang"][row_number]

                }

             entries.append(json.dumps(entry))
                #dict_centroid_phrases_hierarchical = {"main_centroid": "", "entries": entries} # THIS LINE IS FOR COMPARING FORMAT OF RETURNING DICT


        dict_centroid_phrases_hierarchical_subcluster["entries"] = entries

        print("Length of hierarchical subcluster")
        print(len(dict_centroid_phrases_hierarchical_subcluster['entries']))
        print(dict_centroid_phrases_hierarchical_subcluster["main_centroid"])
        print(dict_centroid_phrases_hierarchical_subcluster['entries'])

        dict_subclusters_array_hierarchical.append(json.dumps(dict_centroid_phrases_hierarchical_subcluster))
        # print("dict_centroid_phrases_hierarchical_subcluster")

        print("returned hierarchichal dict")
        print(dict_subclusters_array_hierarchical)

    return dict_subclusters_array_hierarchical, labels


def agglomerative_clustering(descriptors):
    tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+'?\w+\b")
    X = tfidf.fit_transform(descriptors).todense()
    from sklearn.cluster import AgglomerativeClustering


    clustering = AgglomerativeClustering().fit(X)


def k_means_classifier(descriptors, csv_dict_structured_data):

    rng = range(3, 4)
    inertias = []

    # print("type(csv_structured_data)")
    # print(type(csv_structured_data))
    df_structured_data = pd.DataFrame(csv_dict_structured_data) #converting ordered dict to dataframe
    # print("Recieved df_structured_data")
    # print(df_structured_data)
    # print("Recieved descritptors")
    # print(descriptors)
    df_structured_data.fillna('null', inplace=True) # replacing Nan cells with null for js to understand during json parsing

    entries = []
    dict_centroid_phrases = {"main_centroid": "", "entries": entries }
    tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+'?\w+\b")
    X = tfidf.fit_transform(descriptors)
    # print("X tipe of vectorizer")
    # print(type(X))
    # print("X.shape of transformed data")
    # print(X.shape)
    # print("Elements 1 transformed")
    # print(X[1])
    # print("Elements 5 transformed")
    # print(X[6])
    for k in rng:
        model = KMeans(n_clusters=k, random_state=17)
        model.fit(X)
        inertias.append(model.inertia_)

    try:
        draw_classifier_plot(rng, inertias)
    except:
        print('Plotting error')

    # plt.scatter(X,y_kmeans, c=y_kmeans, s=50, cmap='viridis')
    # wcss.append(model.inertia_)
    # plt.plot(range(1, 1), wcss)
    # plt.title('The Elbow Method Graph')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('WCSS')
    # plt.show()
    # print(model.labels_)
    # print(len(model.labels_))

    # print(X.shape)
    # print(len(features_descriptors))
    # print("model.verbose")
    # print(model.verbose)
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = tfidf.get_feature_names()
    # print(len(terms))

    # phrase_numbers = model.cluster_centers_
    # print(phrase_numbers)
    #print(tfidf.decode(phrase_numbers[0]))

    labels = model.labels_  # the model assigns as many labels as rows are in the table
    print("model.labels_")
    print((model.labels_))
    print("model.labels_ LENGHT")
    print(len((model.labels_)))

    all_centroids = []
    for current_cluster in range(len(set(labels))):  # iterates over all unique clusters
        current_neighbors = []
        #print("Synonymicgroup %d:" % i),

        current_centroid = terms[order_centroids[current_cluster, :1][0]]
        all_centroids.append(current_centroid)
        print("Cluster %d:" % current_cluster),
        datapoinst_to_print = 25
        for ind in order_centroids[current_cluster, :datapoinst_to_print]:
            current_neighbors.append(terms[ind])
            # print(terms[ind])

        # print("current_centroid")  # searching for centroids amongst descriptors
        # print(current_centroid)  # searching for centroids amongst descriptors
        for row_number in range(len(csv_dict_structured_data)):
            # print("rows length", len(csv_dict_structured_data))
            # print("labels length", len(labels))
            if ((current_cluster == labels[row_number])):
                # print("df_structured_data[\"ukrlang\"]")
                #print(df_structured_data["ukrlang"])
                print("k =", row_number)
                #print(df_structured_data[k])
                #entries_to_output.append(df_structured_data[k])
                entry = {
                    "centroid": current_centroid,
                    "subcentroid": current_neighbors,
                    "ukrlangphrases": df_structured_data["ukrlang"][row_number],
                    "spanlangphrases": df_structured_data["spanishlang"][row_number],
                    "engllangphrases": df_structured_data["engllang"][row_number],
                    "frenchlangphrases": df_structured_data["frenchlang"][row_number],
                    "itallangphrases": df_structured_data["itallang"][row_number],
                    "latinlangphrases": df_structured_data["latinlang"][row_number],
                    "ruslangphrases": df_structured_data["ruslang"][row_number],
                    "hebrlangphrases": df_structured_data["hebrlang"][row_number]

                }

                entries.append(json.dumps(entry))

    print("all_centroids")
    print(all_centroids)

        #print("current neighbors")
        #print(current_neighbors)

    dict_centroid_phrases["main_centroid"] = max(set(all_centroids), key=all_centroids.count)
    print("dict_centroid_phrases[\"main_centroid\"]")
    print(dict_centroid_phrases["main_centroid"])
    dict_centroid_phrases["entries"] = entries
    return dict_centroid_phrases, labels


def get_k_means_subclusters_array (csv_structured_data, labels):
    print("Lengthes")
    print (len(labels))
    print (len(csv_structured_data))

    clusters_list = []
    for i in set(labels):
        subcluster = [entry for entry, labl in zip(csv_structured_data, labels) if labl == i]
        # print(i, "label" )
        # print("subcluster")
        # print(subcluster)
        clusters_list.append(subcluster)



    return  clusters_list


def k_means_subclusters_classifier (df_subclusters_list, current_language):


    dict_subclusters_array = []
    #print("dataframe", "first element from dataframe")
    #print(df_clusters_list)
    #print(df_clusters_list[0])

    for subcluster in df_subclusters_list:
        descriptors_sequences = []
        #descrs = [entry[current_language+'hetmans'].replace("*", "").strip().lower().split(" ") for entry in subcluster] # storing all descriptors in the given language for each phrase
        for entry in subcluster:
            descr_sequence = entry[current_language+'hetmans'].replace("*", "").strip().lower() # TODO: Perform this normalizing before, for example, in CSV file
            descriptors_sequences.append(descr_sequence)# There should be as many sequences as rows (entries) in subcluster

        #descriptors.extend(descrs)
        dict_subcluster, labels = k_means_classifier(descriptors_sequences, subcluster)
        #print(dict_subcluster)
        dict_subclusters_array.append(json.dumps(dict_subcluster))

    return dict_subclusters_array





app = Flask(__name__)
@app.route('/')
def main():
    dict_from_csv = list(csv.DictReader(open('phrases.csv')))

    current_language = "ukrlang"

    descriptors = [descr[current_language + 'hetmans'] for descr in dict_from_csv]
    dict_centroid_phrases, labels = k_means_classifier(descriptors, dict_from_csv)
    subclusters_dict = get_k_means_subclusters_array(dict_from_csv, labels)
    dict_subcluster_phrases = k_means_subclusters_classifier(subclusters_dict, current_language)

    dict_centroid_phrases_hierarchical, labels2 = hierarchical_clustering(dict_from_csv, current_language)
    return render_template('dict_output.html',
                           dict_centroid_phrases=dict_subcluster_phrases,
                           dict_centroid_phrases_hierarchical=dict_centroid_phrases_hierarchical)

def draw_classifier_plot(rng, inertias):
    plt.plot(rng, inertias, '-o')
    plt.xlabel('number of clusters, k')
    plt.ylabel('inertia')
    plt.xticks(rng)
    plt.show()


@app.route('/get_language', methods=['POST'])
def get_language():

    dict_from_csv = list(csv.DictReader(open('phrases.csv')))
    print("Changing descrpitpors language")
    current_language = request.form.get('select_language')
    if(current_language is None): current_language = "ukrlang"
    print(current_language)
    descriptors = [descr[current_language+'hetmans'] for descr in dict_from_csv]
    dict_centroid_phrases, labels = k_means_classifier(descriptors, dict_from_csv)
    subclusters_dict = get_k_means_subclusters_array(dict_from_csv, labels)
    dict_subcluster_phrases = k_means_subclusters_classifier(subclusters_dict, current_language)

    dict_centroid_phrases_hierarchical, labels2 = hierarchical_clustering(dict_from_csv, current_language)
    return render_template('dict_output.html',
                           dict_centroid_phrases=dict_subcluster_phrases,
                           dict_centroid_phrases_hierarchical=dict_centroid_phrases_hierarchical)



if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)









